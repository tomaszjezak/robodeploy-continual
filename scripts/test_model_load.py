#!/usr/bin/env python3
"""
Quick test to verify PI0 model loads correctly.

This script tests model loading in isolation (no LIBERO required).
Run this first to verify LeRobot + PI0 work before trying the full eval.

Usage:
    python scripts/test_model_load.py
    python scripts/test_model_load.py --model lerobot/act_aloha_sim_transfer_cube_human
"""

import argparse
import sys


def test_imports():
    """Test that all required imports work."""
    print("=" * 60)
    print("Step 1: Testing imports...")
    print("=" * 60)
    
    errors = []
    
    # PyTorch
    try:
        import torch
        cuda_status = "available" if torch.cuda.is_available() else "NOT available"
        print(f"  ✓ PyTorch {torch.__version__} (CUDA: {cuda_status})")
    except ImportError as e:
        errors.append(f"PyTorch: {e}")
        print(f"  ✗ PyTorch: {e}")
    
    # LeRobot
    try:
        import lerobot
        print(f"  ✓ LeRobot {lerobot.__version__}")
    except ImportError as e:
        errors.append(f"LeRobot: {e}")
        print(f"  ✗ LeRobot: {e}")
    
    # LeRobot PI0 policy
    try:
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        print("  ✓ lerobot.policies.pi0.modeling_pi0.PI0Policy")
    except ImportError as e:
        errors.append(f"PI0Policy: {e}")
        print(f"  ✗ PI0Policy: {e}")
    
    # LeRobot ACT policy (backup)
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
        print("  ✓ lerobot.policies.act.modeling_act.ACTPolicy")
    except ImportError as e:
        print(f"  - ACTPolicy (optional): {e}")
    
    # HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download
        print("  ✓ huggingface_hub")
    except ImportError as e:
        errors.append(f"huggingface_hub: {e}")
        print(f"  ✗ huggingface_hub: {e}")
    
    print()
    return len(errors) == 0


def test_model_loading(model_name: str, device: str = "cuda"):
    """Test that the model loads correctly."""
    print("=" * 60)
    print(f"Step 2: Loading model '{model_name}'...")
    print("=" * 60)
    
    import torch
    
    # Determine device
    if device == "cuda" and not torch.cuda.is_available():
        print("  ! CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"  Device: {device}")
    
    try:
        model_lower = model_name.lower()
        
        if "pi0" in model_lower and "pi05" not in model_lower:
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            print("  Loading PI0Policy.from_pretrained()...")
            model = PI0Policy.from_pretrained(model_name)
            
        elif "pi05" in model_lower:
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            print("  Loading PI05Policy.from_pretrained()...")
            model = PI05Policy.from_pretrained(model_name)
            
        elif "act" in model_lower:
            from lerobot.policies.act.modeling_act import ACTPolicy
            print("  Loading ACTPolicy.from_pretrained()...")
            model = ACTPolicy.from_pretrained(model_name)
            
        elif "diffusion" in model_lower:
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
            print("  Loading DiffusionPolicy.from_pretrained()...")
            model = DiffusionPolicy.from_pretrained(model_name)
            
        else:
            print(f"  ✗ Unknown model type: {model_name}")
            return False
        
        # Move to device
        print(f"  Moving model to {device}...")
        model.to(device)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print()
        print(f"  ✓ SUCCESS! Model loaded.")
        print(f"    Type: {type(model).__name__}")
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")
        
        # Memory usage
        if device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"    GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test PI0 model loading")
    parser.add_argument(
        "--model",
        type=str,
        default="lerobot/pi0_base",
        help="Model name on HuggingFace (default: lerobot/pi0_base)",
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
    print("PI0 Model Loading Test")
    print("=" * 60)
    print()
    
    # Test imports
    imports_ok = test_imports()
    if not imports_ok:
        print("Import test FAILED. Fix imports before proceeding.")
        sys.exit(1)
    
    if args.skip_load:
        print("Skipping model load (--skip-load)")
        print()
        print("All import tests PASSED!")
        sys.exit(0)
    
    # Test model loading
    load_ok = test_model_loading(args.model, args.device)
    
    print("=" * 60)
    if load_ok:
        print("All tests PASSED! Model is ready for use.")
        sys.exit(0)
    else:
        print("Model loading FAILED. Check the error above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
