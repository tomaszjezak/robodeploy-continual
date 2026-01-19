#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for PI0.5.

Fine-tunes the base policy on collected deployment data.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for PI0.5")
    parser.add_argument(
        "--task",
        type=str,
        default="libero_long",
        help="LIBERO task suite",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/episodes",
        help="Directory with collected episodes",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/lora_adapters",
        help="Output directory for LoRA weights",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_config.yaml",
        help="LoRA config path",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PI0.5 LoRA Fine-tuning")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Epochs: {args.epochs}")
    print(f"LoRA Rank: {args.lora_rank}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size} x {args.gradient_accumulation} (accum)")
    print()
    
    if args.mock:
        print("Running in MOCK mode")
        
        # Simulate training
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")
            for step in tqdm(range(100), desc="Training"):
                pass
            print(f"  Loss: {0.5 - epoch * 0.05:.4f}")
        
        results = {
            "final_loss": 0.25,
            "epochs": args.epochs,
            "mode": "mock",
        }
    else:
        from src.policy.pi05_wrapper import load_pi05_policy
        from src.training.lora_finetune import LoRATrainer, LoRAConfig, TrainingConfig
        from src.data.episode_buffer import EpisodeBuffer
        
        # Load policy
        print("Loading PI0.5 model...")
        policy = load_pi05_policy()
        
        # Load episode data
        print("Loading episode data...")
        buffer = EpisodeBuffer()
        data_path = Path(args.data_dir) / "episode_buffer.h5"
        if data_path.exists():
            buffer.load(str(data_path))
            episodes = list(buffer)
        else:
            print(f"Warning: No data found at {data_path}")
            print("Collecting fresh episodes...")
            episodes = []  # Would need to collect
        
        if not episodes:
            print("No episodes to train on. Run eval_baseline.py first.")
            return
        
        # Configure LoRA
        lora_config = LoRAConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 4,
        )
        
        training_config = TrainingConfig(
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
        )
        
        # Create trainer
        trainer = LoRATrainer(
            model=policy.model,
            lora_config=lora_config,
            training_config=training_config,
        )
        
        # Train
        results = trainer.train(episodes)
        
        # Save checkpoint
        output_path = Path(args.output_dir) / f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trainer._save_checkpoint(str(output_path))
        results["checkpoint_path"] = str(output_path)
    
    # Print summary
    print()
    print("=" * 60)
    print("Fine-tuning Complete")
    print("=" * 60)
    print(f"Final Loss: {results.get('final_loss', 'N/A')}")
    
    # Save results
    output_path = Path(args.output_dir) / "finetune_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
