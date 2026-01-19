"""
Failure Mining for prioritized data collection.

Identifies and categorizes failure modes for:
- Prioritized replay (focus training on failure cases)
- Correction injection (trigger human/oracle corrections)
- Monitoring (track failure patterns over time)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from .episode_buffer import Episode, FailureCategory


@dataclass
class FailureAnalysis:
    """Analysis result for a single episode."""
    category: FailureCategory
    confidence: float
    details: Dict[str, Any]
    timestep: int  # When failure likely occurred
    suggested_correction: Optional[str] = None


class FailureMiner:
    """
    Analyzes episodes to detect and categorize failure modes.
    
    Uses heuristics based on:
    - Action entropy (uncertainty)
    - Gripper state patterns (grasp failures)
    - Position errors (reach failures)
    - Contact forces (collisions)
    - Episode length (timeouts)
    """
    
    def __init__(
        self,
        entropy_threshold: float = 2.0,
        grasp_slip_threshold: float = 0.1,
        reach_error_threshold: float = 0.05,
        collision_force_threshold: float = 10.0,
        timeout_ratio: float = 0.95,
        max_episode_length: int = 600,
    ):
        """
        Initialize failure miner.
        
        Args:
            entropy_threshold: Entropy above this indicates uncertainty
            grasp_slip_threshold: Gripper delta below this during "holding" = slip
            reach_error_threshold: Position error above this = reach miss
            collision_force_threshold: Force above this = collision
            timeout_ratio: Episode length / max > this = timeout
            max_episode_length: Maximum expected episode length
        """
        self.entropy_threshold = entropy_threshold
        self.grasp_slip_threshold = grasp_slip_threshold
        self.reach_error_threshold = reach_error_threshold
        self.collision_force_threshold = collision_force_threshold
        self.timeout_ratio = timeout_ratio
        self.max_episode_length = max_episode_length
        
        # Track failure patterns over time
        self._failure_history: List[FailureAnalysis] = []
        self._category_counts: Dict[str, int] = defaultdict(int)
    
    def analyze(
        self,
        episode: Episode,
        env_info: Optional[Dict] = None,
    ) -> FailureAnalysis:
        """
        Analyze an episode to determine failure category.
        
        Args:
            episode: Episode to analyze
            env_info: Optional environment info (contact forces, etc.)
            
        Returns:
            FailureAnalysis with category and details
        """
        if episode.success:
            return FailureAnalysis(
                category=FailureCategory.SUCCESS,
                confidence=1.0,
                details={"episode_length": episode.episode_length},
                timestep=-1,
            )
        
        # Run detection heuristics
        detections = []
        
        # Check for timeout
        timeout_result = self._detect_timeout(episode)
        if timeout_result:
            detections.append(timeout_result)
        
        # Check for grasp slip
        grasp_result = self._detect_grasp_slip(episode)
        if grasp_result:
            detections.append(grasp_result)
        
        # Check for reach miss
        reach_result = self._detect_reach_miss(episode)
        if reach_result:
            detections.append(reach_result)
        
        # Check for collision (if env_info available)
        if env_info:
            collision_result = self._detect_collision(episode, env_info)
            if collision_result:
                detections.append(collision_result)
        
        # Check for high entropy (general uncertainty)
        entropy_result = self._detect_high_entropy(episode)
        if entropy_result:
            detections.append(entropy_result)
        
        # Select highest confidence detection
        if detections:
            detections.sort(key=lambda x: x.confidence, reverse=True)
            best = detections[0]
        else:
            best = FailureAnalysis(
                category=FailureCategory.UNKNOWN,
                confidence=0.5,
                details={"reason": "No specific failure pattern detected"},
                timestep=episode.episode_length - 1,
            )
        
        # Update history
        self._failure_history.append(best)
        self._category_counts[best.category.value] += 1
        
        return best
    
    def _detect_timeout(self, episode: Episode) -> Optional[FailureAnalysis]:
        """Detect if episode failed due to timeout."""
        ratio = episode.episode_length / self.max_episode_length
        
        if ratio >= self.timeout_ratio:
            return FailureAnalysis(
                category=FailureCategory.TIMEOUT,
                confidence=ratio,
                details={
                    "episode_length": episode.episode_length,
                    "max_length": self.max_episode_length,
                    "ratio": ratio,
                },
                timestep=episode.episode_length - 1,
                suggested_correction="Increase action efficiency or adjust task timing",
            )
        return None
    
    def _detect_grasp_slip(self, episode: Episode) -> Optional[FailureAnalysis]:
        """Detect if object slipped from gripper."""
        # Assume last element of proprioception is gripper state
        gripper_states = episode.proprioceptions[:, -1]
        
        # Look for pattern: gripper closed (< 0.5) then opens (> 0.5) unexpectedly
        for t in range(1, len(gripper_states)):
            prev = gripper_states[t - 1]
            curr = gripper_states[t]
            
            # Was holding (gripper closed) and suddenly released
            if prev < 0.3 and curr > 0.5:
                # Check if this was intentional release near end
                if t < len(gripper_states) * 0.8:  # Not near end
                    return FailureAnalysis(
                        category=FailureCategory.GRASP_SLIP,
                        confidence=0.8,
                        details={
                            "slip_timestep": t,
                            "gripper_before": float(prev),
                            "gripper_after": float(curr),
                        },
                        timestep=t,
                        suggested_correction="Apply correction to maintain grasp",
                    )
        
        return None
    
    def _detect_reach_miss(self, episode: Episode) -> Optional[FailureAnalysis]:
        """Detect if robot failed to reach target position."""
        # Use action magnitudes as proxy for "still trying to move"
        actions = episode.actions[:, :6]  # Exclude gripper
        action_magnitudes = np.linalg.norm(actions, axis=1)
        
        # If still making large actions at end, probably didn't reach target
        end_window = actions[-20:]  # Last 20 timesteps
        end_magnitude = np.mean(np.linalg.norm(end_window, axis=1))
        
        if end_magnitude > self.reach_error_threshold:
            return FailureAnalysis(
                category=FailureCategory.REACH_MISS,
                confidence=min(end_magnitude / 0.1, 0.9),
                details={
                    "end_action_magnitude": float(end_magnitude),
                    "threshold": self.reach_error_threshold,
                },
                timestep=len(episode.actions) - 10,
                suggested_correction="Guide towards target position",
            )
        
        return None
    
    def _detect_collision(
        self,
        episode: Episode,
        env_info: Dict,
    ) -> Optional[FailureAnalysis]:
        """Detect collision from environment info."""
        contact_forces = env_info.get("contact_forces", [])
        
        if not contact_forces:
            return None
        
        for t, force in enumerate(contact_forces):
            if force > self.collision_force_threshold:
                return FailureAnalysis(
                    category=FailureCategory.COLLISION,
                    confidence=0.95,
                    details={
                        "collision_timestep": t,
                        "contact_force": float(force),
                        "threshold": self.collision_force_threshold,
                    },
                    timestep=t,
                    suggested_correction="Avoid obstacle or adjust trajectory",
                )
        
        return None
    
    def _detect_high_entropy(self, episode: Episode) -> Optional[FailureAnalysis]:
        """Detect failure due to high uncertainty."""
        if episode.entropies is None:
            return None
        
        # Find timesteps with high entropy
        high_entropy_mask = episode.entropies > self.entropy_threshold
        high_entropy_timesteps = np.where(high_entropy_mask)[0]
        
        if len(high_entropy_timesteps) > len(episode.entropies) * 0.3:
            # More than 30% of episode had high entropy
            peak_t = int(np.argmax(episode.entropies))
            return FailureAnalysis(
                category=FailureCategory.OCCLUSION,  # Often caused by visual issues
                confidence=0.6,
                details={
                    "high_entropy_ratio": len(high_entropy_timesteps) / len(episode.entropies),
                    "mean_entropy": float(episode.mean_entropy),
                    "max_entropy": float(episode.max_entropy),
                    "peak_timestep": peak_t,
                },
                timestep=peak_t,
                suggested_correction="Improve observation quality or add training data",
            )
        
        return None
    
    def get_priority_score(self, analysis: FailureAnalysis) -> float:
        """
        Compute priority score for an episode based on failure analysis.
        
        Higher priority = more important for training.
        """
        base_scores = {
            FailureCategory.SUCCESS: 1.0,
            FailureCategory.GRASP_SLIP: 2.0,
            FailureCategory.REACH_MISS: 1.5,
            FailureCategory.COLLISION: 2.5,  # Safety critical
            FailureCategory.TIMEOUT: 1.2,
            FailureCategory.OCCLUSION: 1.5,
            FailureCategory.UNKNOWN: 1.3,
        }
        
        return base_scores[analysis.category] * analysis.confidence
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get failure mining statistics."""
        total = sum(self._category_counts.values())
        
        return {
            "total_analyzed": total,
            "category_counts": dict(self._category_counts),
            "category_ratios": {
                k: v / max(1, total) 
                for k, v in self._category_counts.items()
            },
        }
    
    def get_correction_priority_queue(
        self,
        episodes: List[Episode],
        max_count: int = 50,
    ) -> List[Tuple[Episode, FailureAnalysis]]:
        """
        Get episodes sorted by correction priority.
        
        Returns episodes that would benefit most from corrections.
        """
        analyzed = []
        for ep in episodes:
            if not ep.success:
                analysis = self.analyze(ep)
                priority = self.get_priority_score(analysis)
                analyzed.append((ep, analysis, priority))
        
        # Sort by priority descending
        analyzed.sort(key=lambda x: x[2], reverse=True)
        
        return [(ep, analysis) for ep, analysis, _ in analyzed[:max_count]]
