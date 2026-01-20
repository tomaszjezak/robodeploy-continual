#!/usr/bin/env python3
"""
Quick test to verify PI0.5 model loads correctly.

This script tests model loading in isolation (no LIBERO required).
Run this first to verify OpenPI + PI0.5 work before trying the full eval.

Usage:
    python scripts/test_model_load.py                          # Test OpenPI (default)
    python scripts/test_model_load.py --backend lerobot        # Test LeRobot
    python scripts/test_model_load.py --config pi05_droid      # Different OpenPI config
    python scripts/test_model_load.py --skip-load              # Only test imports
"""

import argparse
import sys


def test_openpi_imports():
    """Test that OpenPI imports work."""
    print("Testing OpenPI imports...")
    errors = []
    
    try:
        from openpi.training import config as openpi_config
        print("  ✓ openpi.training.config")
    except ImportError as e:
        errors.append(f"openpi.training.config: {e}")
        print(f"  ✗ openpi.training.config: {e}")
    
    try:
        from openpi.policies import policy_config
        print("  ✓ openpi.policies.policy_config")
    except ImportError as e:
        errors.append(f"openpi.policies: {e}")
        print(f"  ✗ openpi.policies: {e}")
    
    try:
        from openpi.shared import download
        print("  ✓ openpi.shared.download")
    except ImportError as e:
        errors.append(f"openpi.shared.download: {e}")
        print(f"  ✗ openpi.shared.download: {e}")
    
    return len(errors) == 0


def test_lerobot_imports():
    """Test that LeRobot imports work."""
    print("Testing LeRobot imports...")
    errors = []
    
    try:
        import lerobot
        print(f"  ✓ LeRobot {lerobot.__version__}")
    except ImportError as e:
        errors.append(f"LeRobot: {e}")
        print(f"  ✗ LeRobot: {e}")
    
    try:
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        print("  ✓ lerobot.policies.pi0.modeling_pi0.PI0Policy")
    except ImportError as e:
        errors.append(f"PI0Policy: {e}")
        print(f"  ✗ PI0Policy: {e}")
    
    return len(errors) == 0


def test_common_imports():
    """Test common imports."""
    print("Testing common imports...")
    errors = []
    
    try:
        import torch
        cuda_status = "available" if torch.cuda.is_available() else "NOT available"
        print(f"  ✓ PyTorch {torch.__version__} (CUDA: {cuda_status})")
    except ImportError as e:
        errors.append(f"PyTorch: {e}")
        print(f"  ✗ PyTorch: {e}")
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        errors.append(f"NumPy: {e}")
        print(f"  ✗ NumPy: {e}")
    
    try:
        from huggingface_hub import hf_hub_download
        print("  ✓ huggingface_hub")
    except ImportError as e:
        print(f"  - huggingface_hub (optional): {e}")
    
    return len(errors) == 0


def test_openpi_config(config_name: str):
    """Test that OpenPI config loads."""
    print(f"Testing OpenPI config: {config_name}...")
    
    try:
        from openpi.training import config as openpi_config
        config = openpi_config.get_config(config_name)
        print(f"  ✓ Config loaded: {type(config).__name__}")
        return True
    except Exception as e:
        print(f"  ✗ Config failed: {e}")
        return False


def test_openpi_model_loading(config_name: str, device: str = "cuda"):
    """Test full OpenPI model loading."""
    print(f"Testing OpenPI model loading: {config_name}...")
    
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("  ! CUDA not available, falling back to CPU")
        device = "cpu"
    
    try:
        from openpi.training import config as openpi_config
        from openpi.policies import policy_config
        from openpi.shared import download
        
        print("  Loading config...")
        config = openpi_config.get_config(config_name)
        
        # Build GCS URL for checkpoint download
        checkpoint_url = f"gs://openpi-assets/checkpoints/{config_name}"
        print(f"  Downloading checkpoint from {checkpoint_url}...")
        print("  (This may take several minutes on first run)")
        checkpoint_dir = download.maybe_download(checkpoint_url)
        print(f"  Checkpoint: {checkpoint_dir}")
        
        print("  Creating policy...")
        policy = policy_config.create_trained_policy(config, checkpoint_dir)
        
        print()
        print(f"  ✓ SUCCESS! Policy loaded.")
        print(f"    Type: {type(policy).__name__}")
        
        # Memory usage
        if device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"    GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lerobot_model_loading(model_name: str, device: str = "cuda"):
    """Test LeRobot model loading."""
    print(f"Testing LeRobot model loading: {model_name}...")
    
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("  ! CUDA not available, falling back to CPU")
        device = "cpu"
    
    try:
        model_lower = model_name.lower()
        
        if "pi0" in model_lower and "pi05" not in model_lower:
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            print("  Loading PI0Policy.from_pretrained()...")
            model = PI0Policy.from_pretrained(model_name)
        elif "act" in model_lower:
            from lerobot.policies.act.modeling_act import ACTPolicy
            print("  Loading ACTPolicy.from_pretrained()...")
            model = ACTPolicy.from_pretrained(model_name)
        else:
            print(f"  ✗ Unknown model type: {model_name}")
            return False
        
        model.to(device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        
        print()
        print(f"  ✓ SUCCESS! Model loaded.")
        print(f"    Type: {type(model).__name__}")
        print(f"    Parameters: {total_params:,}")
        
        if device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"    GPU Memory: {allocated:.2f}GB")
        
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test PI0.5 model loading")
    parser.add_argument(
        "--backend",
        type=str,
        default="openpi",
        choices=["openpi", "lerobot"],
        help="Backend to test (default: openpi)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pi05_libero",
        help="OpenPI config name (default: pi05_libero)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lerobot/pi0_base",
        help="LeRobot model name (default: lerobot/pi0_base)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to load model on",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Only test imports, skip model loading",
    )
    
    args = parser.parse_args()
    
    print()
    print("=" * 60)
    print("PI0.5 Model Loading Test")
    print("=" * 60)
    print(f"Backend: {args.backend}")
    print()
    
    # Test common imports
    print("=" * 60)
    print("Step 1: Common imports")
    print("=" * 60)
    common_ok = test_common_imports()
    print()
    
    # Test backend-specific imports
    print("=" * 60)
    print(f"Step 2: {args.backend.upper()} imports")
    print("=" * 60)
    
    if args.backend == "openpi":
        imports_ok = test_openpi_imports()
    else:
        imports_ok = test_lerobot_imports()
    print()
    
    if not imports_ok:
        print("Import test FAILED. Fix imports before proceeding.")
        if args.backend == "openpi":
            print("\nTo install OpenPI:")
            print("  git clone https://github.com/Physical-Intelligence/openpi.git ~/openpi")
            print("  cd ~/openpi && uv sync")
        sys.exit(1)
    
    if args.skip_load:
        print("Skipping model load (--skip-load)")
        print()
        print("All import tests PASSED!")
        sys.exit(0)
    
    # Test config loading (OpenPI only)
    if args.backend == "openpi":
        print("=" * 60)
        print("Step 3: Config loading")
        print("=" * 60)
        config_ok = test_openpi_config(args.config)
        print()
        
        if not config_ok:
            print("Config test FAILED.")
            sys.exit(1)
    
    # Test model loading
    print("=" * 60)
    print(f"Step {'4' if args.backend == 'openpi' else '3'}: Model loading")
    print("=" * 60)
    
    if args.backend == "openpi":
        load_ok = test_openpi_model_loading(args.config, args.device)
    else:
        load_ok = test_lerobot_model_loading(args.model, args.device)
    
    print()
    print("=" * 60)
    if load_ok:
        print("All tests PASSED! Model is ready for use.")
        sys.exit(0)
    else:
        print("Model loading FAILED. Check the error above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
