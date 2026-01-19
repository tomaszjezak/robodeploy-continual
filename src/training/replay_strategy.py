"""
Anti-forgetting strategies for continual learning.

Implements:
- Experience Replay: Mix base task data with deployment data
- EWC (Elastic Weight Consolidation): Regularize important weights
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import copy

from ..data.episode_buffer import Episode, EpisodeBuffer


@dataclass
class ReplayConfig:
    """Configuration for replay-based anti-forgetting."""
    base_task_ratio: float = 0.2  # 20% base task data in each batch
    max_buffer_size: int = 1000
    prioritized: bool = True


class ReplayStrategy:
    """
    Experience replay for anti-forgetting.
    
    Maintains a buffer of base task episodes and mixes them
    with deployment episodes during training.
    """
    
    def __init__(
        self,
        config: Optional[ReplayConfig] = None,
    ):
        """
        Initialize replay strategy.
        
        Args:
            config: Replay configuration
        """
        self.config = config or ReplayConfig()
        self.base_buffer = EpisodeBuffer(
            max_size=self.config.max_buffer_size,
            base_task_ratio=1.0,  # This buffer is all base task
        )
    
    def add_base_episode(self, episode: Episode):
        """Add episode to base task buffer."""
        self.base_buffer.add(episode, is_base_task=True)
    
    def add_base_episodes(self, episodes: List[Episode]):
        """Add multiple base task episodes."""
        for ep in episodes:
            self.add_base_episode(ep)
    
    def get_mixed_batch(
        self,
        deployment_episodes: List[Episode],
        batch_size: int,
    ) -> List[Episode]:
        """
        Get batch mixing deployment and base task episodes.
        
        Args:
            deployment_episodes: Current deployment episodes
            batch_size: Total batch size
            
        Returns:
            Mixed batch of episodes
        """
        n_base = int(batch_size * self.config.base_task_ratio)
        n_deployment = batch_size - n_base
        
        # Sample from deployment
        deployment_sample = []
        if deployment_episodes:
            indices = np.random.choice(
                len(deployment_episodes),
                size=min(n_deployment, len(deployment_episodes)),
                replace=False,
            )
            deployment_sample = [deployment_episodes[i] for i in indices]
        
        # Sample from base buffer
        base_sample = self.base_buffer.sample(
            n_base,
            include_base_task=True,
            prioritized=self.config.prioritized,
        )
        
        # Combine and shuffle
        batch = deployment_sample + base_sample
        np.random.shuffle(batch)
        
        return batch
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get replay buffer statistics."""
        return {
            "base_buffer_size": len(self.base_buffer),
            "config": self.config.__dict__,
        }


@dataclass
class EWCConfig:
    """Configuration for EWC regularization."""
    lambda_ewc: float = 1000.0  # Regularization strength
    sample_size: int = 200  # Episodes to compute Fisher information
    online: bool = False  # Use online EWC variant


class EWCRegularizer:
    """
    Elastic Weight Consolidation for continual learning.
    
    Computes Fisher information to identify important weights,
    then regularizes changes to those weights during new task training.
    
    Reference: "Overcoming catastrophic forgetting in neural networks" (Kirkpatrick et al.)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[EWCConfig] = None,
    ):
        """
        Initialize EWC regularizer.
        
        Args:
            model: Model to regularize
            config: EWC configuration
        """
        self.config = config or EWCConfig()
        self.model = model
        
        # Store Fisher information and optimal parameters
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        
        # For online EWC
        self.task_count = 0
    
    def compute_fisher(
        self,
        episodes: List[Episode],
        device: str = "cuda",
    ):
        """
        Compute Fisher information matrix (diagonal approximation).
        
        Args:
            episodes: Episodes from the task to remember
            device: Device for computation
        """
        # Store current optimal parameters
        self.optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        # Initialize Fisher accumulator
        fisher_accum = {
            name: torch.zeros_like(param.data)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        # Sample episodes
        sample_size = min(self.config.sample_size, len(episodes))
        indices = np.random.choice(len(episodes), size=sample_size, replace=False)
        sampled = [episodes[i] for i in indices]
        
        self.model.eval()
        
        for ep in sampled:
            # Forward pass on episode
            for t in range(len(ep.actions)):
                # Prepare inputs
                image = torch.from_numpy(ep.images[t:t+1]).permute(0, 3, 1, 2).float() / 255.0
                proprio = torch.from_numpy(ep.proprioceptions[t:t+1]).float()
                
                image = image.to(device)
                proprio = proprio.to(device)
                
                # Forward pass (model-specific)
                self.model.zero_grad()
                
                # Get output and compute gradient of log-likelihood
                # This is a simplified version; actual implementation depends on model
                try:
                    output = self.model(image, proprio)
                    if hasattr(output, 'logits'):
                        # Compute gradient of log probability
                        log_probs = torch.log_softmax(output.logits, dim=-1)
                        loss = -log_probs.mean()
                        loss.backward()
                    elif isinstance(output, dict) and 'actions' in output:
                        # For action outputs, use MSE surrogate
                        target = torch.from_numpy(ep.actions[t:t+1]).float().to(device)
                        loss = nn.functional.mse_loss(output['actions'], target)
                        loss.backward()
                except Exception:
                    continue
                
                # Accumulate squared gradients
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher_accum[name] += param.grad.data.pow(2)
        
        # Average and store
        for name in fisher_accum:
            self.fisher[name] = fisher_accum[name] / sample_size
        
        self.task_count += 1
        self.model.train()
    
    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty term.
        
        Returns:
            Penalty to add to loss function
        """
        if not self.fisher:
            return torch.tensor(0.0)
        
        penalty = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for name, param in self.model.named_parameters():
            if name in self.fisher and param.requires_grad:
                # Penalty: lambda * F * (theta - theta*)^2
                diff = param - self.optimal_params[name].to(param.device)
                fisher = self.fisher[name].to(param.device)
                penalty += (fisher * diff.pow(2)).sum()
        
        return self.config.lambda_ewc * penalty
    
    def get_regularized_loss(self, base_loss: torch.Tensor) -> torch.Tensor:
        """
        Add EWC penalty to base loss.
        
        Args:
            base_loss: Original training loss
            
        Returns:
            Loss with EWC regularization
        """
        return base_loss + self.penalty()
    
    def update_online(self, new_fisher: Dict[str, torch.Tensor]):
        """
        Update Fisher information for online EWC.
        
        Combines old and new Fisher information.
        """
        if not self.config.online:
            return
        
        gamma = 1.0 / self.task_count if self.task_count > 0 else 1.0
        
        for name in new_fisher:
            if name in self.fisher:
                # Weighted combination
                self.fisher[name] = (
                    gamma * new_fisher[name] + 
                    (1 - gamma) * self.fisher[name]
                )
            else:
                self.fisher[name] = new_fisher[name]
    
    def save(self, path: str):
        """Save EWC state."""
        torch.save({
            "fisher": self.fisher,
            "optimal_params": self.optimal_params,
            "task_count": self.task_count,
            "config": self.config.__dict__,
        }, path)
    
    def load(self, path: str):
        """Load EWC state."""
        state = torch.load(path)
        self.fisher = state["fisher"]
        self.optimal_params = state["optimal_params"]
        self.task_count = state["task_count"]
