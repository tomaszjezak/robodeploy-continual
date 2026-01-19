"""
Action Chunking for efficient VLA inference.

Implements temporal ensemble and action chunking as used in PI0.5 and similar VLA models.
Reference: OpenVLA-OFT (Optimized Fine-Tuning) for action chunking benefits.
"""

import numpy as np
from typing import Optional, List
from collections import deque


class ActionChunker:
    """
    Action chunking with temporal ensemble for smooth robot control.
    
    Instead of predicting one action at a time, the model predicts a chunk
    of future actions. Temporal ensemble blends overlapping predictions
    for smoother trajectories.
    """
    
    def __init__(
        self,
        chunk_size: int = 16,
        action_dim: int = 7,
        temporal_ensemble: bool = True,
        ensemble_weights: str = "exponential",
        decay_rate: float = 0.5,
    ):
        """
        Initialize action chunker.
        
        Args:
            chunk_size: Number of actions to predict at once
            action_dim: Dimension of action space
            temporal_ensemble: Whether to blend overlapping predictions
            ensemble_weights: "exponential" or "uniform"
            decay_rate: Decay rate for exponential weights
        """
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.temporal_ensemble = temporal_ensemble
        self.ensemble_weights = ensemble_weights
        self.decay_rate = decay_rate
        
        # Buffer for temporal ensemble
        self._prediction_buffer: deque = deque(maxlen=chunk_size)
        self._step_counter = 0
        
        # Precompute weights
        self._weights = self._compute_weights()
    
    def _compute_weights(self) -> np.ndarray:
        """Compute ensemble weights."""
        if self.ensemble_weights == "exponential":
            # Exponential decay: newer predictions weighted more
            weights = np.array([
                self.decay_rate ** i for i in range(self.chunk_size)
            ])
        else:  # uniform
            weights = np.ones(self.chunk_size)
        
        return weights / weights.sum()
    
    def process(self, action_chunk: np.ndarray) -> np.ndarray:
        """
        Process a new action chunk prediction.
        
        Args:
            action_chunk: Predicted actions, shape (chunk_size, action_dim)
            
        Returns:
            Processed actions to execute, shape (chunk_size, action_dim)
        """
        if not self.temporal_ensemble:
            return action_chunk
        
        # Add to buffer
        self._prediction_buffer.append(action_chunk.copy())
        
        # Apply temporal ensemble
        ensembled = self._apply_ensemble()
        
        return ensembled
    
    def _apply_ensemble(self) -> np.ndarray:
        """Apply temporal ensemble to buffered predictions."""
        if len(self._prediction_buffer) == 1:
            return self._prediction_buffer[0]
        
        # Collect all predictions for each timestep
        # Each prediction in buffer has shape (chunk_size, action_dim)
        # We need to align them based on when they were made
        
        result = np.zeros((self.chunk_size, self.action_dim))
        weight_sums = np.zeros(self.chunk_size)
        
        for pred_idx, pred in enumerate(self._prediction_buffer):
            # pred_idx = 0 is the oldest prediction
            # It predicted actions for timesteps [0, chunk_size-1] when it was made
            # Now it's shifted by (len(buffer) - 1 - pred_idx) timesteps
            
            offset = len(self._prediction_buffer) - 1 - pred_idx
            
            for t in range(self.chunk_size):
                target_t = t - offset
                if 0 <= target_t < self.chunk_size:
                    # Weight based on how recent the prediction is
                    weight = self._weights[offset]
                    result[target_t] += weight * pred[t]
                    weight_sums[target_t] += weight
        
        # Normalize
        for t in range(self.chunk_size):
            if weight_sums[t] > 0:
                result[t] /= weight_sums[t]
        
        return result
    
    def get_next_action(self, action_chunk: np.ndarray) -> np.ndarray:
        """
        Get the next single action from a chunk.
        
        This is useful when you want to execute one action at a time
        but still benefit from chunked prediction.
        
        Args:
            action_chunk: Full chunk of predicted actions
            
        Returns:
            Single action to execute, shape (action_dim,)
        """
        # Return the first action in the processed chunk
        processed = self.process(action_chunk)
        return processed[0]
    
    def reset(self):
        """Reset the chunker state."""
        self._prediction_buffer.clear()
        self._step_counter = 0


class ActionBuffer:
    """
    Simple action buffer for storing and replaying action sequences.
    
    Useful for storing expert demonstrations or corrections.
    """
    
    def __init__(self, max_length: int = 1000):
        """
        Initialize action buffer.
        
        Args:
            max_length: Maximum number of actions to store
        """
        self.max_length = max_length
        self._buffer: List[np.ndarray] = []
        self._index = 0
    
    def add(self, action: np.ndarray):
        """Add action to buffer."""
        if len(self._buffer) >= self.max_length:
            self._buffer.pop(0)
        self._buffer.append(action.copy())
    
    def add_chunk(self, actions: np.ndarray):
        """Add multiple actions to buffer."""
        for action in actions:
            self.add(action)
    
    def get(self, index: int) -> Optional[np.ndarray]:
        """Get action at index."""
        if 0 <= index < len(self._buffer):
            return self._buffer[index]
        return None
    
    def get_next(self) -> Optional[np.ndarray]:
        """Get next action in sequence."""
        if self._index < len(self._buffer):
            action = self._buffer[self._index]
            self._index += 1
            return action
        return None
    
    def reset_playback(self):
        """Reset playback index to start."""
        self._index = 0
    
    def clear(self):
        """Clear all stored actions."""
        self._buffer.clear()
        self._index = 0
    
    def __len__(self) -> int:
        return len(self._buffer)
    
    def to_numpy(self) -> np.ndarray:
        """Convert buffer to numpy array."""
        if not self._buffer:
            return np.array([])
        return np.stack(self._buffer)
