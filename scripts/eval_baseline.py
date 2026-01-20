#!/usr/bin/env python3
"""
Baseline Evaluation Script for PI0.5 on LIBERO.

Runs the base policy and collects metrics:
- Success rate
- Action entropy
- Failure categories
- Episode statistics
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm


def create_libero_env(task: str = "libero_spatial", task_id: int = 0):
    """Create LIBERO environment."""
    try:
        import os
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        
        # Get task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task]()
        
        # Get specific task
        task_spec = task_suite.get_task(task_id)
        task_name = task_spec.name
        
        # Construct FULL path to bddl file (including benchmark subdirectory)
        bddl_dir = get_libero_path("bddl_files")
        bddl_full_path = os.path.join(bddl_dir, task, task_spec.bddl_file)
        
        print(f"BDDL path: {bddl_full_path}")
        print(f"Exists: {os.path.exists(bddl_full_path)}")
        
        # Create environment
        env_args = {
            "bddl_file_name": bddl_full_path,
            "camera_heights": 256,
            "camera_widths": 256,
        }
        
        env = OffScreenRenderEnv(**env_args)
        env.task_name = task_name
        
        return env, task_name
        
    except ImportError as e:
        print(f"Error importing LIBERO: {e}")
        print("Please install LIBERO: pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git")
        return None, None


def load_policy(config_path: str = "configs/pi05_config.yaml"):
    """Load PI0.5 policy."""
    from src.policy.pi05_wrapper import PI05Policy
    
    policy = PI05Policy(config_path)
    policy.load()  # Policy handles mock mode internally if loading fails
    
    return policy


def run_evaluation(
    env,
    policy,
    n_episodes: int = 50,
    max_steps: int = 600,
    verbose: bool = True,
):
    """
    Run evaluation episodes.
    
    Returns:
        dict: Evaluation results
    """
    from src.data.episode_buffer import Episode, FailureCategory
    from src.data.failure_mining import FailureMiner
    
    results = {
        "episodes": [],
        "success_count": 0,
        "total_episodes": 0,
        "failure_categories": {},
        "entropies": [],
        "episode_lengths": [],
    }
    
    failure_miner = FailureMiner()
    
    for ep_idx in tqdm(range(n_episodes), desc="Evaluating", disable=not verbose):
        # Reset
        obs = env.reset()
        policy.reset()
        
        images = []
        proprios = []
        actions = []
        entropies = []
        
        done = False
        success = False
        step = 0
        
        while not done and step < max_steps:
            # Get observation
            image = obs.get("agentview_image", np.zeros((256, 256, 3), dtype=np.uint8))
            proprio = obs.get("robot0_proprio-state", np.zeros(8))
            
            # Get action
            output = policy.get_action(image, proprio)
            
            # Store
            images.append(image)
            proprios.append(proprio)
            actions.append(output.action)
            entropies.append(output.entropy)
            
            # Step
            obs, reward, done, info = env.step(output.action)
            step += 1
            
            if info.get("success", False):
                success = True
                done = True
        
        # Create episode
        episode = Episode(
            images=np.array(images),
            proprioceptions=np.array(proprios),
            actions=np.array(actions),
            success=success,
            entropies=np.array(entropies),
        )
        
        # Analyze failure
        analysis = failure_miner.analyze(episode)
        episode.failure_category = analysis.category
        
        # Record results
        results["episodes"].append({
            "success": success,
            "length": step,
            "mean_entropy": float(np.mean(entropies)) if entropies else 0,
            "failure_category": analysis.category.value,
        })
        
        if success:
            results["success_count"] += 1
        else:
            cat = analysis.category.value
            results["failure_categories"][cat] = results["failure_categories"].get(cat, 0) + 1
        
        results["entropies"].extend(entropies)
        results["episode_lengths"].append(step)
        results["total_episodes"] += 1
    
    # Compute summary
    results["success_rate"] = results["success_count"] / results["total_episodes"]
    results["mean_entropy"] = float(np.mean(results["entropies"])) if results["entropies"] else 0
    results["mean_episode_length"] = float(np.mean(results["episode_lengths"]))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PI0.5 baseline on LIBERO")
    parser.add_argument(
        "--task",
        type=str,
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90", "libero_100"],
        help="LIBERO task suite",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="Task ID within suite (0-9)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pi05_config.yaml",
        help="Policy config path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/baseline_eval.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (no actual model/env)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PI0.5 Baseline Evaluation on LIBERO")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Task ID: {args.task_id}")
    print(f"Episodes: {args.n_episodes}")
    print()
    
    if args.mock:
        print("Running in MOCK mode - no actual evaluation")
        results = {
            "success_rate": 0.65,
            "mean_entropy": 1.5,
            "mean_episode_length": 300,
            "total_episodes": args.n_episodes,
            "success_count": int(args.n_episodes * 0.65),
            "failure_categories": {
                "reach_miss": 8,
                "grasp_slip": 5,
                "timeout": 4,
            },
            "mode": "mock",
        }
    else:
        # Create environment
        env, task_name = create_libero_env(args.task, args.task_id)
        if env is None:
            print("Failed to create environment. Use --mock for testing.")
            return
        
        print(f"Task name: {task_name}")
        
        # Load policy
        policy = load_policy(args.config)
        
        # Run evaluation
        results = run_evaluation(
            env=env,
            policy=policy,
            n_episodes=args.n_episodes,
        )
        
        env.close()
    
    # Add metadata
    results["timestamp"] = datetime.now().isoformat()
    results["task"] = args.task
    results["task_id"] = args.task_id
    results["config"] = args.config
    
    # Print summary
    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Mean Entropy: {results['mean_entropy']:.2f}")
    print(f"Mean Episode Length: {results['mean_episode_length']:.0f}")
    print()
    print("Failure Categories:")
    for cat, count in results.get("failure_categories", {}).items():
        print(f"  {cat}: {count}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
