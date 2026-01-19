"""
SOP-style Online Continual Learning Orchestrator.

Implements the main continual learning loop:
1. Robot runs policy, collects data
2. Monitor triggers update based on reliability thresholds
3. Learner fine-tunes on collected data + corrections
4. Sync updated weights back to robot

Reference: SOP (Scalable Online Post-training) - arXiv 2601.03044
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
from datetime import datetime
from threading import Thread
import queue

from ..policy.pi05_wrapper import PI05Policy
from ..data.episode_buffer import EpisodeBuffer, Episode, FailureCategory
from ..data.failure_mining import FailureMiner, FailureAnalysis
from ..data.corrections import CorrectionGenerator, HybridCorrector, CorrectedEpisode
from ..training.lora_finetune import LoRATrainer, LoRAConfig, TrainingConfig
from ..training.replay_strategy import ReplayStrategy, EWCRegularizer
from .metrics_monitor import MetricsMonitor, ReliabilityMetrics
from .weight_sync import WeightSynchronizer


@dataclass
class OnlineConfig:
    """Configuration for online continual learning."""
    # Update triggers
    update_every_n_episodes: int = 20
    min_episodes_for_update: int = 10
    success_rate_threshold: float = 0.90
    entropy_threshold: float = 2.0
    consecutive_failure_threshold: int = 3
    
    # Training composition (HG-DAgger style)
    on_policy_ratio: float = 0.8
    correction_ratio: float = 0.2
    
    # Training per update
    gradient_steps_per_update: int = 50
    
    # Async settings
    async_updates: bool = False
    
    # Paths
    checkpoint_dir: str = "checkpoints/lora_adapters"
    log_dir: str = "logs"
    
    # Flywheel tracking
    max_cycles: int = 20
    target_success_rate: float = 0.95


@dataclass
class CycleResult:
    """Result of one learning cycle."""
    cycle_id: int
    episodes_collected: int
    success_rate_before: float
    success_rate_after: float
    update_triggered: bool
    training_loss: Optional[float] = None
    timestamp: str = ""


class OnlineLearner:
    """
    Main orchestrator for SOP-style online continual learning.
    
    Coordinates:
    - Policy execution and data collection
    - Reliability monitoring and update triggering
    - LoRA fine-tuning with corrections
    - Weight synchronization
    """
    
    def __init__(
        self,
        policy: PI05Policy,
        env,  # LIBERO environment
        config: Optional[OnlineConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        correction_generator: Optional[CorrectionGenerator] = None,
    ):
        """
        Initialize online learner.
        
        Args:
            policy: PI0.5 policy to fine-tune
            env: LIBERO environment for rollouts
            config: Online learning configuration
            lora_config: LoRA configuration
            correction_generator: Correction generator for HG-DAgger
        """
        self.policy = policy
        self.env = env
        self.config = config or OnlineConfig()
        self.lora_config = lora_config or LoRAConfig()
        
        # Components
        self.episode_buffer = EpisodeBuffer(
            max_size=500,
            base_task_ratio=0.2,
        )
        self.failure_miner = FailureMiner()
        self.correction_generator = correction_generator or HybridCorrector(env)
        self.metrics_monitor = MetricsMonitor(
            success_threshold=self.config.success_rate_threshold,
            entropy_threshold=self.config.entropy_threshold,
            window_size=50,
        )
        self.weight_sync = WeightSynchronizer(
            checkpoint_dir=self.config.checkpoint_dir,
        )
        self.replay_strategy = ReplayStrategy()
        
        # Trainer (initialized lazily)
        self._trainer: Optional[LoRATrainer] = None
        
        # State
        self.cycle_count = 0
        self.total_episodes = 0
        self.update_count = 0
        self.cycle_history: List[CycleResult] = []
        
        # Async queue for updates
        self._update_queue: queue.Queue = queue.Queue()
        self._async_thread: Optional[Thread] = None
        
        # Setup directories
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
    
    def run(
        self,
        num_cycles: Optional[int] = None,
        callback: Optional[Callable[[CycleResult], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the online learning loop.
        
        Args:
            num_cycles: Number of cycles to run (default: config.max_cycles)
            callback: Called after each cycle with results
            
        Returns:
            Summary of learning run
        """
        num_cycles = num_cycles or self.config.max_cycles
        
        print(f"Starting online continual learning for {num_cycles} cycles")
        print(f"Target success rate: {self.config.target_success_rate:.1%}")
        
        # Start async update thread if configured
        if self.config.async_updates:
            self._start_async_thread()
        
        try:
            for cycle in range(num_cycles):
                result = self._run_cycle(cycle)
                self.cycle_history.append(result)
                
                # Callback
                if callback:
                    callback(result)
                
                # Log progress
                self._log_cycle(result)
                
                # Check if target reached
                if result.success_rate_after >= self.config.target_success_rate:
                    print(f"Target success rate reached! ({result.success_rate_after:.1%})")
                    break
                
        finally:
            if self.config.async_updates:
                self._stop_async_thread()
        
        return self._generate_summary()
    
    def _run_cycle(self, cycle_id: int) -> CycleResult:
        """Run a single learning cycle."""
        self.cycle_count = cycle_id
        
        # Get metrics before
        metrics_before = self.metrics_monitor.get_current_metrics()
        success_rate_before = metrics_before.success_rate if metrics_before else 0.0
        
        # Collect episodes
        episodes = self._collect_episodes(self.config.update_every_n_episodes)
        
        # Analyze failures
        analyses = []
        for ep in episodes:
            analysis = self.failure_miner.analyze(ep)
            ep.failure_category = analysis.category
            ep.compute_priority(self.episode_buffer.priority_config)
            analyses.append(analysis)
        
        # Add to buffer
        for ep in episodes:
            self.episode_buffer.add(ep)
        
        # Check if update should trigger
        update_triggered = self._should_update()
        training_loss = None
        
        if update_triggered:
            # Generate corrections for failures
            failed_episodes = [ep for ep in episodes if not ep.success]
            failed_analyses = [a for ep, a in zip(episodes, analyses) if not ep.success]
            
            corrections = []
            if failed_episodes:
                for ep, analysis in zip(failed_episodes, failed_analyses):
                    corr = self.correction_generator.generate_corrections(ep, analysis)
                    corrections.extend(corr)
            
            # Run training update
            if self.config.async_updates:
                self._queue_update(episodes, corrections)
            else:
                training_loss = self._run_update(episodes, corrections)
            
            self.update_count += 1
        
        # Get metrics after
        metrics_after = self.metrics_monitor.get_current_metrics()
        success_rate_after = metrics_after.success_rate if metrics_after else success_rate_before
        
        return CycleResult(
            cycle_id=cycle_id,
            episodes_collected=len(episodes),
            success_rate_before=success_rate_before,
            success_rate_after=success_rate_after,
            update_triggered=update_triggered,
            training_loss=training_loss,
            timestamp=datetime.now().isoformat(),
        )
    
    def _collect_episodes(self, n_episodes: int) -> List[Episode]:
        """Collect episodes by running policy in environment."""
        episodes = []
        
        for i in range(n_episodes):
            episode = self._run_episode()
            episodes.append(episode)
            
            # Update metrics
            self.metrics_monitor.add_episode(episode)
            self.total_episodes += 1
            
            # Check for immediate intervention
            if self.metrics_monitor.check_consecutive_failures(
                self.config.consecutive_failure_threshold
            ):
                print(f"Warning: {self.config.consecutive_failure_threshold} consecutive failures!")
        
        return episodes
    
    def _run_episode(self) -> Episode:
        """Run single episode and collect data."""
        # Reset environment
        obs = self.env.reset()
        self.policy.reset()
        
        # Storage
        images = []
        proprios = []
        actions = []
        entropies = []
        
        done = False
        success = False
        step = 0
        
        while not done:
            # Extract observation components
            image = obs.get("agentview_image", obs.get("image", np.zeros((256, 256, 3))))
            proprio = obs.get("robot0_proprio-state", np.zeros(8))
            
            # Get action from policy
            output = self.policy.get_action(
                image=image,
                proprioception=proprio,
                language_instruction=self.env.get_task_description() if hasattr(self.env, 'get_task_description') else "complete the task",
            )
            
            # Store data
            images.append(image)
            proprios.append(proprio)
            actions.append(output.action)
            entropies.append(output.entropy)
            
            # Step environment
            obs, reward, done, info = self.env.step(output.action)
            step += 1
            
            # Check success
            if info.get("success", False):
                success = True
                done = True
        
        return Episode(
            images=np.array(images),
            proprioceptions=np.array(proprios),
            actions=np.array(actions),
            success=success,
            entropies=np.array(entropies),
            task_name=self.env.task_name if hasattr(self.env, 'task_name') else "unknown",
        )
    
    def _should_update(self) -> bool:
        """Check if training update should be triggered."""
        metrics = self.metrics_monitor.get_current_metrics()
        
        if metrics is None:
            return False
        
        # Trigger if success rate below threshold
        if metrics.success_rate < self.config.success_rate_threshold:
            return True
        
        # Trigger if mean entropy too high
        if metrics.mean_entropy > self.config.entropy_threshold:
            return True
        
        # Trigger based on episode count
        if self.total_episodes % self.config.update_every_n_episodes == 0:
            episodes_since_update = self.total_episodes - (self.update_count * self.config.update_every_n_episodes)
            if episodes_since_update >= self.config.min_episodes_for_update:
                return True
        
        return False
    
    def _run_update(
        self,
        episodes: List[Episode],
        corrections: List,
    ) -> float:
        """Run training update."""
        print(f"Running training update (episodes: {len(episodes)}, corrections: {len(corrections)})")
        
        # Initialize trainer if needed
        if self._trainer is None:
            self._trainer = LoRATrainer(
                model=self.policy.model,
                lora_config=self.lora_config,
                training_config=TrainingConfig(
                    num_epochs=1,
                    max_steps=self.config.gradient_steps_per_update,
                ),
            )
        
        # Get mixed batch with replay
        train_episodes = self.replay_strategy.get_mixed_batch(
            episodes,
            batch_size=len(episodes),
        )
        
        # Train
        result = self._trainer.train(train_episodes)
        
        # Sync weights
        lora_weights = self._trainer.get_lora_weights()
        self.weight_sync.save_weights(
            lora_weights,
            f"update_{self.update_count}",
        )
        
        return result.get("final_loss", 0.0)
    
    def _queue_update(self, episodes: List[Episode], corrections: List):
        """Queue update for async processing."""
        self._update_queue.put((episodes, corrections))
    
    def _start_async_thread(self):
        """Start async update thread."""
        def worker():
            while True:
                try:
                    item = self._update_queue.get(timeout=1.0)
                    if item is None:
                        break
                    episodes, corrections = item
                    self._run_update(episodes, corrections)
                except queue.Empty:
                    continue
        
        self._async_thread = Thread(target=worker, daemon=True)
        self._async_thread.start()
    
    def _stop_async_thread(self):
        """Stop async update thread."""
        if self._async_thread:
            self._update_queue.put(None)
            self._async_thread.join(timeout=5.0)
    
    def _log_cycle(self, result: CycleResult):
        """Log cycle result."""
        print(f"Cycle {result.cycle_id}: "
              f"success {result.success_rate_before:.1%} → {result.success_rate_after:.1%}, "
              f"episodes: {result.episodes_collected}, "
              f"update: {result.update_triggered}")
        
        # Save to log file
        log_path = Path(self.config.log_dir) / "cycle_history.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(result.__dict__) + "\n")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate learning summary."""
        if not self.cycle_history:
            return {"status": "no_cycles_completed"}
        
        initial_rate = self.cycle_history[0].success_rate_before
        final_rate = self.cycle_history[-1].success_rate_after
        
        return {
            "total_cycles": len(self.cycle_history),
            "total_episodes": self.total_episodes,
            "total_updates": self.update_count,
            "initial_success_rate": initial_rate,
            "final_success_rate": final_rate,
            "improvement": final_rate - initial_rate,
            "target_reached": final_rate >= self.config.target_success_rate,
            "failure_statistics": self.failure_miner.get_statistics(),
            "buffer_statistics": self.episode_buffer.get_statistics(),
        }
    
    def save_state(self, path: str):
        """Save learner state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save episode buffer
        self.episode_buffer.save(path / "episode_buffer.h5")
        
        # Save cycle history
        with open(path / "cycle_history.json", "w") as f:
            json.dump([r.__dict__ for r in self.cycle_history], f, indent=2)
        
        # Save metrics
        self.metrics_monitor.save(path / "metrics.json")
    
    def load_state(self, path: str):
        """Load learner state."""
        path = Path(path)
        
        # Load episode buffer
        buffer_path = path / "episode_buffer.h5"
        if buffer_path.exists():
            self.episode_buffer.load(buffer_path)
        
        # Load cycle history
        history_path = path / "cycle_history.json"
        if history_path.exists():
            with open(history_path) as f:
                data = json.load(f)
                self.cycle_history = [CycleResult(**d) for d in data]
        
        # Load metrics
        metrics_path = path / "metrics.json"
        if metrics_path.exists():
            self.metrics_monitor.load(metrics_path)
