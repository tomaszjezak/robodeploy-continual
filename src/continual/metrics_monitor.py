"""
Metrics Monitoring for reliability tracking.

Tracks:
- Success rate over sliding window
- Action entropy distribution
- Failure categories
- Update triggers
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path
from datetime import datetime

from ..data.episode_buffer import Episode, FailureCategory


@dataclass
class ReliabilityMetrics:
    """Current reliability metrics snapshot."""
    success_rate: float
    episode_count: int
    mean_entropy: float
    max_entropy: float
    failure_distribution: Dict[str, int]
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class TriggerEvent:
    """Record of an update trigger."""
    trigger_type: str  # "success_rate", "entropy", "consecutive_failures"
    value: float
    threshold: float
    timestamp: str = ""
    episode_id: int = 0


class MetricsMonitor:
    """
    Monitor reliability metrics and determine update triggers.
    
    Implements the "long march of 9s" tracking:
    - 90% → 99% → 99.9% reliability improvement
    """
    
    def __init__(
        self,
        success_threshold: float = 0.90,
        entropy_threshold: float = 2.0,
        window_size: int = 50,
    ):
        """
        Initialize metrics monitor.
        
        Args:
            success_threshold: Success rate below this triggers update
            entropy_threshold: Entropy above this flags as pre-fail
            window_size: Sliding window size for metrics
        """
        self.success_threshold = success_threshold
        self.entropy_threshold = entropy_threshold
        self.window_size = window_size
        
        # Sliding windows
        self._success_window: deque = deque(maxlen=window_size)
        self._entropy_window: deque = deque(maxlen=window_size)
        self._failure_window: deque = deque(maxlen=window_size)
        
        # History
        self._metrics_history: List[ReliabilityMetrics] = []
        self._trigger_history: List[TriggerEvent] = []
        
        # Cumulative stats
        self._total_episodes = 0
        self._total_successes = 0
        self._failure_counts: Dict[str, int] = {cat.value: 0 for cat in FailureCategory}
        
        # Consecutive failure tracking
        self._consecutive_failures = 0
    
    def add_episode(self, episode: Episode):
        """
        Add episode to monitoring.
        
        Args:
            episode: Completed episode
        """
        self._total_episodes += 1
        
        # Update success tracking
        self._success_window.append(1 if episode.success else 0)
        if episode.success:
            self._total_successes += 1
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            self._failure_counts[episode.failure_category.value] += 1
        
        # Update entropy tracking
        if episode.entropies is not None and len(episode.entropies) > 0:
            self._entropy_window.append(episode.mean_entropy)
        
        # Update failure category
        self._failure_window.append(episode.failure_category.value)
        
        # Snapshot metrics periodically
        if self._total_episodes % 10 == 0:
            metrics = self.get_current_metrics()
            if metrics:
                self._metrics_history.append(metrics)
    
    def get_current_metrics(self) -> Optional[ReliabilityMetrics]:
        """Get current metrics snapshot."""
        if not self._success_window:
            return None
        
        # Compute success rate
        success_rate = sum(self._success_window) / len(self._success_window)
        
        # Compute entropy stats
        mean_entropy = np.mean(self._entropy_window) if self._entropy_window else 0.0
        max_entropy = max(self._entropy_window) if self._entropy_window else 0.0
        
        # Failure distribution (from window)
        failure_dist = {}
        for cat in FailureCategory:
            count = sum(1 for f in self._failure_window if f == cat.value)
            if count > 0:
                failure_dist[cat.value] = count
        
        return ReliabilityMetrics(
            success_rate=success_rate,
            episode_count=self._total_episodes,
            mean_entropy=float(mean_entropy),
            max_entropy=float(max_entropy),
            failure_distribution=failure_dist,
        )
    
    def check_success_threshold(self) -> Optional[TriggerEvent]:
        """Check if success rate is below threshold."""
        if not self._success_window:
            return None
        
        success_rate = sum(self._success_window) / len(self._success_window)
        
        if success_rate < self.success_threshold:
            event = TriggerEvent(
                trigger_type="success_rate",
                value=success_rate,
                threshold=self.success_threshold,
                timestamp=datetime.now().isoformat(),
                episode_id=self._total_episodes,
            )
            self._trigger_history.append(event)
            return event
        
        return None
    
    def check_entropy_threshold(self) -> Optional[TriggerEvent]:
        """Check if entropy is above threshold."""
        if not self._entropy_window:
            return None
        
        mean_entropy = np.mean(self._entropy_window)
        
        if mean_entropy > self.entropy_threshold:
            event = TriggerEvent(
                trigger_type="entropy",
                value=float(mean_entropy),
                threshold=self.entropy_threshold,
                timestamp=datetime.now().isoformat(),
                episode_id=self._total_episodes,
            )
            self._trigger_history.append(event)
            return event
        
        return None
    
    def check_consecutive_failures(self, threshold: int = 3) -> bool:
        """Check if there are too many consecutive failures."""
        if self._consecutive_failures >= threshold:
            event = TriggerEvent(
                trigger_type="consecutive_failures",
                value=float(self._consecutive_failures),
                threshold=float(threshold),
                timestamp=datetime.now().isoformat(),
                episode_id=self._total_episodes,
            )
            self._trigger_history.append(event)
            return True
        return False
    
    def get_reliability_level(self) -> str:
        """
        Get current reliability level ("march of 9s").
        
        Returns:
            Level string: "low", "90%", "99%", "99.9%", "99.99%"
        """
        if not self._success_window:
            return "unknown"
        
        success_rate = sum(self._success_window) / len(self._success_window)
        
        if success_rate >= 0.9999:
            return "99.99%"
        elif success_rate >= 0.999:
            return "99.9%"
        elif success_rate >= 0.99:
            return "99%"
        elif success_rate >= 0.90:
            return "90%"
        else:
            return "low"
    
    def get_improvement_trajectory(self) -> List[Tuple[int, float]]:
        """Get success rate over time."""
        return [
            (m.episode_count, m.success_rate)
            for m in self._metrics_history
        ]
    
    def get_trigger_history(self) -> List[TriggerEvent]:
        """Get history of trigger events."""
        return self._trigger_history
    
    def get_failure_analysis(self) -> Dict[str, Any]:
        """Get detailed failure analysis."""
        total_failures = sum(self._failure_counts.values()) - self._failure_counts.get("success", 0)
        
        if total_failures == 0:
            return {"status": "no_failures"}
        
        # Category percentages
        category_pcts = {
            cat: count / max(1, total_failures) * 100
            for cat, count in self._failure_counts.items()
            if cat != "success" and count > 0
        }
        
        # Most common
        sorted_cats = sorted(category_pcts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_failures": total_failures,
            "total_episodes": self._total_episodes,
            "failure_rate": total_failures / max(1, self._total_episodes),
            "category_percentages": category_pcts,
            "most_common": sorted_cats[0][0] if sorted_cats else None,
            "counts": self._failure_counts,
        }
    
    def save(self, path: str):
        """Save metrics to file."""
        data = {
            "total_episodes": self._total_episodes,
            "total_successes": self._total_successes,
            "failure_counts": self._failure_counts,
            "metrics_history": [m.__dict__ for m in self._metrics_history],
            "trigger_history": [t.__dict__ for t in self._trigger_history],
            "config": {
                "success_threshold": self.success_threshold,
                "entropy_threshold": self.entropy_threshold,
                "window_size": self.window_size,
            },
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load metrics from file."""
        with open(path) as f:
            data = json.load(f)
        
        self._total_episodes = data["total_episodes"]
        self._total_successes = data["total_successes"]
        self._failure_counts = data["failure_counts"]
        
        self._metrics_history = [
            ReliabilityMetrics(**m) for m in data["metrics_history"]
        ]
        self._trigger_history = [
            TriggerEvent(**t) for t in data["trigger_history"]
        ]


# Type alias for tuple
from typing import Tuple
