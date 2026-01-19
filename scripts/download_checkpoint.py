#!/usr/bin/env python3
"""
Download PI0.5 checkpoint from HuggingFace.
"""

import os
import argparse
from pathlib import Path


def download_checkpoint(
    model_name: str = "TensorAuto/tPi0.5-libero",
    output_dir: str = "checkpoints",
    use_lerobot: bool = True,
):
    """
    Download PI0.5 checkpoint.
    
    Args:
        model_name: HuggingFace model ID
        output_dir: Local directory to save
        use_lerobot: Use LeRobot for downloading (recommended)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading checkpoint: {model_name}")
    print(f"Output directory: {output_dir}")
    
    if use_lerobot:
        try:
            from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
            from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
            
            print("Using LeRobot to download...")
            
            # This will download to HuggingFace cache
            config = PI0Config.from_pretrained(model_name)
            model = PI0Policy.from_pretrained(model_name, config=config)
            
            # Save locally
            local_path = output_dir / model_name.replace("/", "_")
            model.save_pretrained(local_path)
            
            print(f"Saved to: {local_path}")
            return str(local_path)
            
        except ImportError:
            print("LeRobot not available, falling back to huggingface_hub")
        except Exception as e:
            print(f"LeRobot download failed: {e}")
            print("Falling back to huggingface_hub")
    
    # Fallback: use huggingface_hub directly
    try:
        from huggingface_hub import snapshot_download
        
        print("Using huggingface_hub to download...")
        
        local_path = snapshot_download(
            repo_id=model_name,
            local_dir=str(output_dir / model_name.replace("/", "_")),
            local_dir_use_symlinks=False,
        )
        
        print(f"Downloaded to: {local_path}")
        return local_path
        
    except ImportError:
        print("ERROR: huggingface_hub not installed")
        print("Install with: pip install huggingface-hub")
        return None
    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        return None


def download_libero_dataset(
    dataset_name: str = "openvla/modified_libero_rlds",
    output_dir: str = "data",
):
    """
    Download pre-converted LIBERO dataset.
    
    Args:
        dataset_name: HuggingFace dataset ID
        output_dir: Local directory to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset: {dataset_name}")
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset(dataset_name, split="train")
        
        # Save locally
        local_path = output_dir / dataset_name.replace("/", "_")
        dataset.save_to_disk(str(local_path))
        
        print(f"Saved to: {local_path}")
        print(f"Dataset size: {len(dataset)} examples")
        return str(local_path)
        
    except ImportError:
        print("ERROR: datasets not installed")
        print("Install with: pip install datasets")
        return None
    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download PI0.5 checkpoint")
    parser.add_argument(
        "--model",
        type=str,
        default="TensorAuto/tPi0.5-libero",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory",
    )
    parser.add_argument(
        "--download-dataset",
        action="store_true",
        help="Also download LIBERO dataset",
    )
    
    args = parser.parse_args()
    
    # Download checkpoint
    download_checkpoint(args.model, args.output_dir)
    
    # Optionally download dataset
    if args.download_dataset:
        download_libero_dataset()


if __name__ == "__main__":
    main()
