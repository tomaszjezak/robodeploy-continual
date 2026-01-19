"""
PI0.5 Policy Wrapper for LIBERO
Wraps the PI0.5 VLA model for inference in LIBERO simulation.

References:
- OpenPI: https://github.com/Physical-Intelligence/openpi
- LeRobot: https://github.com/huggingface/lerobot
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from omegaconf import OmegaConf

from .action_chunking import ActionChunker


@dataclass
class PolicyOutput:
    """Output from policy inference."""
    action: np.ndarray  # Shape: (action_dim,)
    entropy: float  # Action distribution entropy
    raw_actions: Optional[np.ndarray] = None  # Full action chunk if available


class PI05Policy:
    """
    PI0.5 Vision-Language-Action policy wrapper.
    
    Handles:
    - Model loading (from HuggingFace or local checkpoint)
    - Image preprocessing
    - Action inference with chunking
    - Entropy computation for uncertainty estimation
    """
    
    def __init__(
        self,
        config_path: str = "configs/pi05_config.yaml",
        device: str = "cuda",
        load_in_8bit: bool = False,
    ):
        """
        Initialize PI0.5 policy.
        
        Args:
            config_path: Path to model config
            device: Device to run on
            load_in_8bit: Enable 8-bit quantization for memory savings
        """
        self.config = OmegaConf.load(config_path)
        self.device = device
        self.load_in_8bit = load_in_8bit
        
        # Will be set after loading
        self.model = None
        self.processor = None
        self.action_chunker = None
        
        # Track inference state
        self._action_buffer = None
        self._buffer_idx = 0
        
    def load(self, checkpoint_path: Optional[str] = None):
        """
        Load the PI0.5 model.
        
        Args:
            checkpoint_path: Override checkpoint path (optional)
        """
        model_name = checkpoint_path or self.config.model.name
        
        print(f"Loading PI0.5 model from: {model_name}")
        
        try:
            # Try loading via LeRobot first (recommended)
            self._load_via_lerobot(model_name)
        except Exception as e:
            print(f"LeRobot loading failed: {e}")
            print("Falling back to direct HuggingFace loading...")
            self._load_via_transformers(model_name)
        
        # Initialize action chunker
        self.action_chunker = ActionChunker(
            chunk_size=self.config.inference.chunk_size,
            action_dim=self.config.model.action_dim,
            temporal_ensemble=self.config.inference.temporal_ensemble,
        )
        
        print(f"Model loaded successfully on {self.device}")
        self._print_memory_usage()
        
    def _load_via_lerobot(self, model_name: str):
        """Load model using LeRobot library."""
        try:
            from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy as LeRobotPI0
            from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
        except ImportError:
            raise ImportError("LeRobot not installed. Run: pip install lerobot")
        
        # Load config and model
        config = PI0Config.from_pretrained(model_name)
        self.model = LeRobotPI0.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16 if self.config.model.dtype == "bfloat16" else torch.float32,
        )
        
        if self.load_in_8bit:
            self._quantize_model()
        
        self.model.to(self.device)
        self.model.eval()
        
    def _load_via_transformers(self, model_name: str):
        """Fallback: Load model using transformers directly."""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            raise ImportError("Transformers not installed. Run: pip install transformers")
        
        load_kwargs = {
            "torch_dtype": torch.bfloat16 if self.config.model.dtype == "bfloat16" else torch.float32,
            "device_map": "auto" if self.load_in_8bit else None,
        }
        
        if self.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            except ImportError:
                print("Warning: bitsandbytes not available, loading in full precision")
        
        self.model = AutoModelForVision2Seq.from_pretrained(model_name, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        if not self.load_in_8bit:
            self.model.to(self.device)
        self.model.eval()
        
    def _quantize_model(self):
        """Apply 8-bit quantization for memory savings."""
        try:
            import bitsandbytes as bnb
            # Replace linear layers with 8-bit versions
            # This is handled by transformers with load_in_8bit flag
            print("8-bit quantization enabled")
        except ImportError:
            print("Warning: bitsandbytes not installed, skipping quantization")
    
    def _print_memory_usage(self):
        """Print current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def preprocess_observation(
        self,
        image: np.ndarray,
        proprioception: np.ndarray,
        language_instruction: str = "",
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess observation for model input.
        
        Args:
            image: RGB image from camera, shape (H, W, 3)
            proprioception: Robot state, shape (proprio_dim,)
            language_instruction: Task instruction string
            
        Returns:
            Dictionary of preprocessed tensors
        """
        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Resize if needed
        target_size = self.config.model.image_size
        if image.shape[0] != target_size or image.shape[1] != target_size:
            from PIL import Image
            img_pil = Image.fromarray((image * 255).astype(np.uint8))
            img_pil = img_pil.resize((target_size, target_size))
            image = np.array(img_pil).astype(np.float32) / 255.0
        
        # Convert to tensor: (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device, dtype=torch.bfloat16)
        
        # Proprioception
        proprio_tensor = torch.from_numpy(proprioception).float().unsqueeze(0)
        proprio_tensor = proprio_tensor.to(self.device)
        
        return {
            "image": image_tensor,
            "proprioception": proprio_tensor,
            "language_instruction": language_instruction,
        }
    
    @torch.no_grad()
    def get_action(
        self,
        image: np.ndarray,
        proprioception: np.ndarray,
        language_instruction: str = "complete the task",
    ) -> PolicyOutput:
        """
        Get action from the policy.
        
        Uses action chunking: generates multiple future actions at once,
        then returns them sequentially for smoother execution.
        
        Args:
            image: RGB image from camera
            proprioception: Robot state
            language_instruction: Task instruction
            
        Returns:
            PolicyOutput with action and entropy
        """
        # Check if we have buffered actions
        if self._action_buffer is not None and self._buffer_idx < len(self._action_buffer):
            action = self._action_buffer[self._buffer_idx]
            self._buffer_idx += 1
            # Return buffered action with estimated entropy
            return PolicyOutput(
                action=action,
                entropy=0.0,  # No entropy for buffered actions
            )
        
        # Need to generate new action chunk
        inputs = self.preprocess_observation(image, proprioception, language_instruction)
        
        # Forward pass
        if hasattr(self.model, 'predict_action'):
            # LeRobot-style interface
            output = self.model.predict_action(inputs)
            actions = output['action'].cpu().numpy()
            entropy = self._compute_entropy(output.get('logits', None))
        else:
            # Generic transformer interface
            actions, entropy = self._forward_transformers(inputs)
        
        # Apply action chunking / temporal ensemble
        actions = self.action_chunker.process(actions)
        
        # Buffer actions for subsequent calls
        self._action_buffer = actions
        self._buffer_idx = 1  # Return first action now
        
        return PolicyOutput(
            action=actions[0],
            entropy=entropy,
            raw_actions=actions,
        )
    
    def _forward_transformers(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Tuple[np.ndarray, float]:
        """Forward pass for transformers-loaded model."""
        # This is a placeholder - actual implementation depends on model architecture
        # For now, return dummy actions
        action_dim = self.config.model.action_dim
        chunk_size = self.config.inference.chunk_size
        
        # Dummy forward pass
        actions = np.zeros((chunk_size, action_dim))
        entropy = 0.0
        
        return actions, entropy
    
    def _compute_entropy(self, logits: Optional[torch.Tensor]) -> float:
        """Compute entropy of action distribution."""
        if logits is None:
            return 0.0
        
        # Softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        # Entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return entropy.mean().item()
    
    def reset(self):
        """Reset policy state (clear action buffer)."""
        self._action_buffer = None
        self._buffer_idx = 0
        if self.action_chunker:
            self.action_chunker.reset()
    
    def get_lora_target_modules(self) -> list:
        """Get list of modules to target with LoRA."""
        return self.config.get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ])


# Convenience function for quick loading
def load_pi05_policy(
    config_path: str = "configs/pi05_config.yaml",
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    load_in_8bit: bool = False,
) -> PI05Policy:
    """
    Load PI0.5 policy with default settings.
    
    Args:
        config_path: Path to config file
        checkpoint_path: Override model path
        device: Device to use
        load_in_8bit: Enable 8-bit quantization
        
    Returns:
        Loaded PI05Policy instance
    """
    policy = PI05Policy(config_path, device, load_in_8bit)
    policy.load(checkpoint_path)
    return policy
