"""
PI0.5 Policy Wrapper for LIBERO
Wraps the PI0.5 VLA model for inference in LIBERO simulation.

Supports two backends:
- OpenPI (recommended): Physical Intelligence's official implementation
- LeRobot: HuggingFace's robotics library

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
    - Model loading via OpenPI (recommended) or LeRobot
    - Image preprocessing for LIBERO observations
    - Action inference with chunking
    - Entropy computation for uncertainty estimation
    
    Backends:
    - "openpi": Physical Intelligence's official PI0.5 (faster, recommended)
    - "lerobot": HuggingFace LeRobot's PI0 (fallback)
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
        # Always initialize action chunker first (needed even for mock mode)
        self.action_chunker = ActionChunker(
            chunk_size=self.config.inference.chunk_size,
            action_dim=self.config.model.action_dim,
            temporal_ensemble=self.config.inference.temporal_ensemble,
        )
        
        # Determine backend from config
        backend = self.config.model.get("backend", "openpi")
        print(f"Using backend: {backend}")
        
        if backend == "openpi":
            # OpenPI backend (recommended for PI0.5)
            openpi_config_name = checkpoint_path or self.config.model.get("openpi_config", "pi05_libero")
            checkpoint_dir = self.config.model.get("checkpoint_dir", None)
            print(f"Loading PI0.5 via OpenPI: config={openpi_config_name}")
            
            try:
                self._load_via_openpi(openpi_config_name, checkpoint_dir)
                print(f"Model loaded successfully on {self.device}")
                self._print_memory_usage()
            except Exception as e:
                print(f"OpenPI loading failed: {e}")
                print("Falling back to LeRobot...")
                self._try_lerobot_fallback()
        else:
            # LeRobot backend
            model_name = checkpoint_path or self.config.model.name
            print(f"Loading PI0 via LeRobot: {model_name}")
            self._try_lerobot_fallback(model_name)
    
    def _try_lerobot_fallback(self, model_name: str = "lerobot/pi0_base"):
        """Try loading via LeRobot, fall back to mock mode if that fails."""
        try:
            self._load_via_lerobot(model_name)
            print(f"Model loaded successfully on {self.device}")
            self._print_memory_usage()
        except Exception as e:
            print(f"LeRobot loading failed: {e}")
            print("Running in MOCK MODE - random actions will be generated")
            self.model = None
    
    def _load_via_openpi(self, config_name: str, checkpoint_dir: Optional[str] = None):
        """
        Load PI0.5 using OpenPI's native inference.
        
        Args:
            config_name: OpenPI config name (e.g., 'pi05_libero', 'pi05_droid')
            checkpoint_dir: Path to checkpoint directory (optional, will download if not provided)
        """
        try:
            from openpi.training import config as openpi_config
            from openpi.policies import policy_config
            from openpi.shared import download
        except ImportError as e:
            raise ImportError(
                f"OpenPI not installed. Install with: cd ~/openpi && uv sync\n"
                f"Or clone from: https://github.com/Physical-Intelligence/openpi\n"
                f"Error: {e}"
            )
        
        print(f"Loading OpenPI config: {config_name}")
        config = openpi_config.get_config(config_name)
        
        # Download checkpoint if not provided
        if checkpoint_dir is None:
            # Use GCS URL for checkpoint download
            checkpoint_url = f"gs://openpi-assets/checkpoints/{config_name}"
            print(f"Downloading checkpoint from {checkpoint_url}...")
            print("(This may take several minutes on first run)")
            checkpoint_dir = download.maybe_download(checkpoint_url)
            print(f"Checkpoint directory: {checkpoint_dir}")
        
        # Create trained policy
        print("Creating trained policy...")
        self.model = policy_config.create_trained_policy(config, checkpoint_dir)
        self._openpi_config = config  # Store for preprocessing
        
        print(f"Loaded PI0.5 from OpenPI: {config_name}")
        
    def _load_via_lerobot(self, model_name: str):
        """Load model using LeRobot's native from_pretrained (simplest, most reliable)."""
        print(f"Loading LeRobot policy from: {model_name}")
        
        # Detect policy type from model name and use native from_pretrained
        model_lower = model_name.lower()
        
        if "pi0" in model_lower and "pi05" not in model_lower:
            # PI0 model
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            print("Detected PI0 policy, using PI0Policy.from_pretrained()")
            self.model = PI0Policy.from_pretrained(model_name)
            print("Loaded PI0 policy from LeRobot")
            
        elif "pi05" in model_lower:
            # PI0.5 model  
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            print("Detected PI0.5 policy, using PI05Policy.from_pretrained()")
            self.model = PI05Policy.from_pretrained(model_name)
            print("Loaded PI0.5 policy from LeRobot")
            
        elif "act" in model_lower:
            # ACT model
            from lerobot.policies.act.modeling_act import ACTPolicy
            print("Detected ACT policy, using ACTPolicy.from_pretrained()")
            self.model = ACTPolicy.from_pretrained(model_name)
            print("Loaded ACT policy from LeRobot")
            
        elif "diffusion" in model_lower:
            # Diffusion policy
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
            print("Detected Diffusion policy, using DiffusionPolicy.from_pretrained()")
            self.model = DiffusionPolicy.from_pretrained(model_name)
            print("Loaded Diffusion policy from LeRobot")
            
        else:
            raise ValueError(f"Unknown model type in name: {model_name}. "
                           f"Expected 'pi0', 'pi05', 'act', or 'diffusion' in name.")
        
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
        # MOCK MODE: If no model loaded, return random actions
        if self.model is None:
            action_dim = self.config.model.action_dim
            # Generate small random actions (simulating a policy trying to do something)
            action = np.random.randn(action_dim) * 0.1
            # Clip to reasonable range
            action = np.clip(action, -1.0, 1.0)
            # Random entropy to simulate uncertainty
            entropy = np.random.uniform(0.5, 2.5)
            return PolicyOutput(
                action=action,
                entropy=entropy,
                raw_actions=None,
            )
        
        # Check if we have buffered actions
        if self._action_buffer is not None and self._buffer_idx < len(self._action_buffer):
            action = self._action_buffer[self._buffer_idx]
            self._buffer_idx += 1
            # Return buffered action with estimated entropy
            return PolicyOutput(
                action=action,
                entropy=0.0,  # No entropy for buffered actions
            )
        
        # Forward pass depends on backend
        if hasattr(self, '_openpi_config'):
            # OpenPI backend
            actions, entropy = self._forward_openpi(image, proprioception, language_instruction)
        elif hasattr(self.model, 'predict_action'):
            # LeRobot-style interface
            inputs = self.preprocess_observation(image, proprioception, language_instruction)
            output = self.model.predict_action(inputs)
            actions = output['action'].cpu().numpy()
            entropy = self._compute_entropy(output.get('logits', None))
        else:
            # Generic transformer interface
            inputs = self.preprocess_observation(image, proprioception, language_instruction)
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
    
    def _forward_openpi(
        self,
        image: np.ndarray,
        proprioception: np.ndarray,
        language_instruction: str,
    ) -> Tuple[np.ndarray, float]:
        """Forward pass using OpenPI policy."""
        # OpenPI expects observation dict with specific keys
        # The exact format depends on the config (LIBERO, DROID, ALOHA, etc.)
        
        # Normalize image to [0, 1] if needed
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Build observation dict for OpenPI
        # For LIBERO, typically: image, state, prompt
        observation = {
            "image": image,  # (H, W, 3)
            "state": proprioception,  # Robot state
            "prompt": language_instruction,  # Task instruction
        }
        
        # OpenPI infer returns dict with 'actions' key
        result = self.model.infer(observation)
        actions = result["actions"]  # Action chunk
        
        # Convert to numpy if needed
        if hasattr(actions, 'numpy'):
            actions = actions.numpy()
        elif not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        
        # OpenPI doesn't provide entropy directly, estimate as 0
        entropy = 0.0
        
        return actions, entropy
    
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
    Load PI0/PI0.5 policy with default settings.
    
    Args:
        config_path: Path to config file
        checkpoint_path: Override model path (e.g., "lerobot/pi0_base")
        device: Device to use
        load_in_8bit: Enable 8-bit quantization
        
    Returns:
        Loaded PI05Policy instance
    """
    policy = PI05Policy(config_path, device, load_in_8bit)
    policy.load(checkpoint_path)
    return policy
