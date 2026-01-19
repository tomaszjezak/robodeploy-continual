"""
Weight Synchronization for policy updates.

Handles:
- Local file-based weight sync
- Optional HuggingFace Hub sync for "cloud" simulation
"""

import torch
import json
from typing import Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import shutil


class WeightSynchronizer:
    """
    Synchronize model weights between learner and robot.
    
    Supports:
    - Local file swap (default)
    - HuggingFace Hub push (for cloud simulation)
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/lora_adapters",
        sync_method: str = "local",  # "local" or "huggingface"
        hf_repo_id: Optional[str] = None,
    ):
        """
        Initialize weight synchronizer.
        
        Args:
            checkpoint_dir: Local directory for checkpoints
            sync_method: "local" or "huggingface"
            hf_repo_id: HuggingFace repo ID (if using HF sync)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.sync_method = sync_method
        self.hf_repo_id = hf_repo_id
        
        # Track versions
        self._current_version = 0
        self._version_history: list = []
        
        # Latest checkpoint pointer
        self._latest_path = self.checkpoint_dir / "latest"
    
    def save_weights(
        self,
        weights: Dict[str, torch.Tensor],
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Save weights to checkpoint.
        
        Args:
            weights: Dictionary of weight tensors
            name: Checkpoint name (default: versioned)
            metadata: Optional metadata to save
            
        Returns:
            Path to saved checkpoint
        """
        self._current_version += 1
        name = name or f"v{self._current_version}"
        
        save_path = self.checkpoint_dir / name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save weights
        torch.save(weights, save_path / "adapter_weights.pt")
        
        # Save metadata
        meta = {
            "version": self._current_version,
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "num_tensors": len(weights),
            "total_params": sum(w.numel() for w in weights.values()),
            **(metadata or {}),
        }
        
        with open(save_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        # Update latest pointer
        self._update_latest(save_path)
        
        # Track in history
        self._version_history.append({
            "version": self._current_version,
            "path": str(save_path),
            "timestamp": meta["timestamp"],
        })
        
        # Sync to cloud if configured
        if self.sync_method == "huggingface":
            self._push_to_hub(save_path)
        
        return str(save_path)
    
    def load_weights(
        self,
        path: Optional[str] = None,
        version: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Load weights from checkpoint.
        
        Args:
            path: Specific path to load from
            version: Specific version to load
            
        Returns:
            Dictionary of weight tensors
        """
        if path:
            load_path = Path(path)
        elif version:
            load_path = self.checkpoint_dir / f"v{version}"
        else:
            # Load latest
            load_path = self._get_latest()
        
        if not load_path or not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        
        weights = torch.load(load_path / "adapter_weights.pt")
        return weights
    
    def load_latest(self) -> Optional[Dict[str, torch.Tensor]]:
        """Load latest weights, return None if no checkpoint exists."""
        try:
            return self.load_weights()
        except FileNotFoundError:
            return None
    
    def _update_latest(self, path: Path):
        """Update the 'latest' symlink/pointer."""
        # Write path to latest file
        with open(self._latest_path, "w") as f:
            f.write(str(path))
    
    def _get_latest(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        if not self._latest_path.exists():
            return None
        
        with open(self._latest_path) as f:
            return Path(f.read().strip())
    
    def _push_to_hub(self, path: Path):
        """Push checkpoint to HuggingFace Hub."""
        if not self.hf_repo_id:
            print("Warning: No HF repo ID configured, skipping push")
            return
        
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            api.upload_folder(
                folder_path=str(path),
                repo_id=self.hf_repo_id,
                repo_type="model",
                path_in_repo=f"checkpoints/{path.name}",
            )
            print(f"Pushed checkpoint to HuggingFace: {self.hf_repo_id}")
            
        except ImportError:
            print("Warning: huggingface_hub not installed, skipping push")
        except Exception as e:
            print(f"Warning: Failed to push to HuggingFace: {e}")
    
    def pull_from_hub(self, version: Optional[str] = None) -> str:
        """
        Pull checkpoint from HuggingFace Hub.
        
        Args:
            version: Specific version to pull (default: latest)
            
        Returns:
            Local path to downloaded checkpoint
        """
        if not self.hf_repo_id:
            raise ValueError("No HF repo ID configured")
        
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            
            # Download entire checkpoints folder
            local_path = snapshot_download(
                repo_id=self.hf_repo_id,
                allow_patterns=["checkpoints/*"],
                local_dir=str(self.checkpoint_dir.parent),
            )
            
            return local_path
            
        except ImportError:
            raise ImportError("huggingface_hub required for cloud sync")
    
    def get_version_history(self) -> list:
        """Get history of saved versions."""
        return self._version_history
    
    def get_current_version(self) -> int:
        """Get current version number."""
        return self._current_version
    
    def cleanup_old_versions(self, keep_last: int = 5):
        """
        Remove old checkpoints, keeping only recent ones.
        
        Args:
            keep_last: Number of recent checkpoints to keep
        """
        if len(self._version_history) <= keep_last:
            return
        
        to_remove = self._version_history[:-keep_last]
        
        for entry in to_remove:
            path = Path(entry["path"])
            if path.exists():
                shutil.rmtree(path)
        
        self._version_history = self._version_history[-keep_last:]
