"""
Episode Buffer for storing and managing rollout data.

Supports:
- Prioritized replay based on failure categories and entropy
- Efficient storage and retrieval
- HDF5 serialization for persistence
"""

import numpy as np
import h5py
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Iterator
from enum import Enum
import random
from collections import deque


class FailureCategory(str, Enum):
    """Categorical failure modes for prioritization."""
    SUCCESS = "success"
    GRASP_SLIP = "grasp_slip"
    REACH_MISS = "reach_miss"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    OCCLUSION = "occlusion"
    UNKNOWN = "unknown"


@dataclass
class Episode:
    """
    Single episode of robot interaction.
    
    Stores observations, actions, and metadata for replay/training.
    """
    # Core data
    images: np.ndarray  # Shape: (T, H, W, C)
    proprioceptions: np.ndarray  # Shape: (T, proprio_dim)
    actions: np.ndarray  # Shape: (T, action_dim)
    
    # Outcome
    success: bool
    failure_category: FailureCategory = FailureCategory.SUCCESS
    
    # Metadata
    episode_length: int = 0
    task_name: str = ""
    language_instruction: str = ""
    
    # Uncertainty metrics
    entropies: Optional[np.ndarray] = None  # Shape: (T,)
    mean_entropy: float = 0.0
    max_entropy: float = 0.0
    
    # Priority for replay
    priority: float = 1.0
    
    # Timestamps
    timestamp: str = ""
    
    def __post_init__(self):
        """Compute derived fields."""
        self.episode_length = len(self.actions)
        
        if self.entropies is not None:
            self.mean_entropy = float(np.mean(self.entropies))
            self.max_entropy = float(np.max(self.entropies))
        
        if not self.success:
            if self.failure_category == FailureCategory.SUCCESS:
                self.failure_category = FailureCategory.UNKNOWN
    
    def compute_priority(self, config: Dict[str, float]) -> float:
        """
        Compute priority score for replay sampling.
        
        Args:
            config: Priority weights for different factors
            
        Returns:
            Priority score (higher = more important)
        """
        base_priority = 1.0
        
        # Failures are more important
        if not self.success:
            failure_weights = config.get("failure_weights", {})
            weight = failure_weights.get(self.failure_category.value, 1.5)
            base_priority *= weight
        
        # High entropy = uncertain = important
        entropy_weight = config.get("entropy_weight", 1.5)
        entropy_threshold = config.get("entropy_threshold", 2.0)
        if self.mean_entropy > entropy_threshold:
            base_priority *= entropy_weight
        
        self.priority = base_priority
        return self.priority


class EpisodeBuffer:
    """
    Buffer for storing episodes with prioritized replay support.
    
    Features:
    - Priority-based sampling (failures and high-entropy episodes weighted higher)
    - Separate tracking for base task vs deployment episodes
    - HDF5 persistence
    - Efficient memory management
    """
    
    def __init__(
        self,
        max_size: int = 500,
        base_task_ratio: float = 0.2,
        priority_config: Optional[Dict] = None,
    ):
        """
        Initialize episode buffer.
        
        Args:
            max_size: Maximum number of episodes to store
            base_task_ratio: Ratio of base task episodes to maintain
            priority_config: Configuration for priority computation
        """
        self.max_size = max_size
        self.base_task_ratio = base_task_ratio
        self.priority_config = priority_config or {
            "failure_weights": {
                "grasp_slip": 2.0,
                "reach_miss": 1.5,
                "collision": 2.5,
                "timeout": 1.0,
                "occlusion": 1.5,
                "unknown": 1.2,
            },
            "entropy_weight": 1.5,
            "entropy_threshold": 2.0,
        }
        
        # Separate buffers
        self._deployment_buffer: deque = deque(maxlen=max_size)
        self._base_task_buffer: deque = deque(maxlen=int(max_size * base_task_ratio))
        
        # Statistics
        self._stats = {
            "total_added": 0,
            "successes": 0,
            "failures": 0,
            "failure_categories": {cat.value: 0 for cat in FailureCategory},
        }
    
    def add(self, episode: Episode, is_base_task: bool = False):
        """
        Add episode to buffer.
        
        Args:
            episode: Episode to add
            is_base_task: Whether this is from base task (for anti-forgetting)
        """
        # Compute priority
        episode.compute_priority(self.priority_config)
        
        # Add to appropriate buffer
        if is_base_task:
            self._base_task_buffer.append(episode)
        else:
            self._deployment_buffer.append(episode)
        
        # Update stats
        self._stats["total_added"] += 1
        if episode.success:
            self._stats["successes"] += 1
        else:
            self._stats["failures"] += 1
            self._stats["failure_categories"][episode.failure_category.value] += 1
    
    def sample(
        self,
        batch_size: int,
        include_base_task: bool = True,
        prioritized: bool = True,
    ) -> List[Episode]:
        """
        Sample episodes from buffer.
        
        Args:
            batch_size: Number of episodes to sample
            include_base_task: Whether to include base task episodes
            prioritized: Use priority-weighted sampling
            
        Returns:
            List of sampled episodes
        """
        # Determine how many from each buffer
        if include_base_task and len(self._base_task_buffer) > 0:
            n_base = int(batch_size * self.base_task_ratio)
            n_deployment = batch_size - n_base
        else:
            n_base = 0
            n_deployment = batch_size
        
        # Sample from deployment buffer
        deployment_samples = self._sample_from_buffer(
            self._deployment_buffer, n_deployment, prioritized
        )
        
        # Sample from base task buffer
        base_samples = self._sample_from_buffer(
            self._base_task_buffer, n_base, prioritized
        )
        
        # Combine and shuffle
        samples = deployment_samples + base_samples
        random.shuffle(samples)
        
        return samples
    
    def _sample_from_buffer(
        self,
        buffer: deque,
        n: int,
        prioritized: bool,
    ) -> List[Episode]:
        """Sample n episodes from a buffer."""
        if len(buffer) == 0 or n == 0:
            return []
        
        n = min(n, len(buffer))
        
        if not prioritized:
            # Uniform sampling
            indices = random.sample(range(len(buffer)), n)
        else:
            # Priority-weighted sampling
            priorities = np.array([ep.priority for ep in buffer])
            probs = priorities / priorities.sum()
            indices = np.random.choice(len(buffer), size=n, replace=False, p=probs)
        
        return [buffer[i] for i in indices]
    
    def get_failure_episodes(
        self,
        category: Optional[FailureCategory] = None,
        max_count: int = 100,
    ) -> List[Episode]:
        """Get episodes that failed, optionally filtered by category."""
        failures = [
            ep for ep in self._deployment_buffer
            if not ep.success and (category is None or ep.failure_category == category)
        ]
        return failures[:max_count]
    
    def get_high_entropy_episodes(
        self,
        threshold: float = 2.0,
        max_count: int = 100,
    ) -> List[Episode]:
        """Get episodes with high action entropy (uncertain)."""
        high_entropy = [
            ep for ep in self._deployment_buffer
            if ep.mean_entropy > threshold
        ]
        # Sort by entropy descending
        high_entropy.sort(key=lambda ep: ep.mean_entropy, reverse=True)
        return high_entropy[:max_count]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            **self._stats,
            "deployment_buffer_size": len(self._deployment_buffer),
            "base_task_buffer_size": len(self._base_task_buffer),
            "success_rate": self._stats["successes"] / max(1, self._stats["total_added"]),
        }
    
    def save(self, path: str):
        """Save buffer to HDF5 file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(path, "w") as f:
            # Save deployment episodes
            deploy_grp = f.create_group("deployment")
            self._save_episodes_to_group(deploy_grp, list(self._deployment_buffer))
            
            # Save base task episodes
            base_grp = f.create_group("base_task")
            self._save_episodes_to_group(base_grp, list(self._base_task_buffer))
            
            # Save metadata
            f.attrs["stats"] = json.dumps(self._stats)
            f.attrs["max_size"] = self.max_size
            f.attrs["base_task_ratio"] = self.base_task_ratio
    
    def _save_episodes_to_group(self, group: h5py.Group, episodes: List[Episode]):
        """Save list of episodes to HDF5 group."""
        group.attrs["count"] = len(episodes)
        
        for i, ep in enumerate(episodes):
            ep_grp = group.create_group(f"episode_{i}")
            ep_grp.create_dataset("images", data=ep.images, compression="gzip")
            ep_grp.create_dataset("proprioceptions", data=ep.proprioceptions)
            ep_grp.create_dataset("actions", data=ep.actions)
            
            if ep.entropies is not None:
                ep_grp.create_dataset("entropies", data=ep.entropies)
            
            # Metadata as attributes
            ep_grp.attrs["success"] = ep.success
            ep_grp.attrs["failure_category"] = ep.failure_category.value
            ep_grp.attrs["task_name"] = ep.task_name
            ep_grp.attrs["language_instruction"] = ep.language_instruction
            ep_grp.attrs["priority"] = ep.priority
    
    def load(self, path: str):
        """Load buffer from HDF5 file."""
        with h5py.File(path, "r") as f:
            # Load metadata
            self._stats = json.loads(f.attrs["stats"])
            self.max_size = f.attrs["max_size"]
            self.base_task_ratio = f.attrs["base_task_ratio"]
            
            # Load deployment episodes
            self._deployment_buffer.clear()
            deploy_grp = f["deployment"]
            for i in range(deploy_grp.attrs["count"]):
                ep = self._load_episode_from_group(deploy_grp[f"episode_{i}"])
                self._deployment_buffer.append(ep)
            
            # Load base task episodes
            self._base_task_buffer.clear()
            base_grp = f["base_task"]
            for i in range(base_grp.attrs["count"]):
                ep = self._load_episode_from_group(base_grp[f"episode_{i}"])
                self._base_task_buffer.append(ep)
    
    def _load_episode_from_group(self, group: h5py.Group) -> Episode:
        """Load episode from HDF5 group."""
        entropies = None
        if "entropies" in group:
            entropies = np.array(group["entropies"])
        
        return Episode(
            images=np.array(group["images"]),
            proprioceptions=np.array(group["proprioceptions"]),
            actions=np.array(group["actions"]),
            success=group.attrs["success"],
            failure_category=FailureCategory(group.attrs["failure_category"]),
            task_name=group.attrs["task_name"],
            language_instruction=group.attrs["language_instruction"],
            entropies=entropies,
            priority=group.attrs["priority"],
        )
    
    def __len__(self) -> int:
        return len(self._deployment_buffer) + len(self._base_task_buffer)
    
    def __iter__(self) -> Iterator[Episode]:
        yield from self._deployment_buffer
        yield from self._base_task_buffer
