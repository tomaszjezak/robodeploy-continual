#!/usr/bin/env python3
"""
Run the full SOP-style online continual learning loop.

This is the main entry point for the continual learning system.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run SOP-style continual learning")
    parser.add_argument(
        "--task",
        type=str,
        default="libero_long",
        help="LIBERO task suite",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="Task ID within suite",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=10,
        help="Number of learning cycles",
    )
    parser.add_argument(
        "--episodes-per-cycle",
        type=int,
        default=20,
        help="Episodes per cycle",
    )
    parser.add_argument(
        "--shift-level",
        type=str,
        default="medium",
        choices=["none", "low", "medium", "high"],
        help="Domain shift level",
    )
    parser.add_argument(
        "--target-success",
        type=float,
        default=0.95,
        help="Target success rate",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Config directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs",
        help="Output directory for logs",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SOP-style Online Continual Learning")
    print("=" * 60)
    print(f"Task: {args.task} (ID: {args.task_id})")
    print(f"Cycles: {args.cycles}")
    print(f"Episodes/cycle: {args.episodes_per_cycle}")
    print(f"Domain shift: {args.shift_level}")
    print(f"Target success: {args.target_success:.0%}")
    print()
    
    if args.mock:
        print("Running in MOCK mode")
        print()
        
        # Simulate learning loop
        cycle_history = []
        success_rate = 0.60  # Start at 60%
        
        for cycle in range(args.cycles):
            # Simulate improvement
            if success_rate < args.target_success:
                improvement = 0.05 + (args.target_success - success_rate) * 0.1
                success_rate = min(success_rate + improvement, args.target_success)
            
            result = {
                "cycle_id": cycle,
                "episodes_collected": args.episodes_per_cycle,
                "success_rate_before": success_rate - 0.02,
                "success_rate_after": success_rate,
                "update_triggered": success_rate < 0.90,
                "timestamp": datetime.now().isoformat(),
            }
            cycle_history.append(result)
            
            print(f"Cycle {cycle}: "
                  f"success {result['success_rate_before']:.1%} → {result['success_rate_after']:.1%}")
            
            if success_rate >= args.target_success:
                print(f"\nTarget reached! ({success_rate:.1%})")
                break
        
        summary = {
            "total_cycles": len(cycle_history),
            "initial_success_rate": 0.60,
            "final_success_rate": success_rate,
            "improvement": success_rate - 0.60,
            "target_reached": success_rate >= args.target_success,
            "mode": "mock",
        }
        
    else:
        from src.policy.pi05_wrapper import load_pi05_policy
        from src.continual.online_learner import OnlineLearner, OnlineConfig
        from src.continual.dashboard import Dashboard
        from scripts.eval_baseline import create_libero_env
        
        # Create environment
        print("Creating LIBERO environment...")
        env, task_name = create_libero_env(args.task, args.task_id)
        if env is None:
            print("Failed to create environment. Use --mock for testing.")
            return
        
        # Load policy
        print("Loading PI0.5 model...")
        policy = load_pi05_policy()
        
        # Configure online learner
        config = OnlineConfig(
            update_every_n_episodes=args.episodes_per_cycle,
            max_cycles=args.cycles,
            target_success_rate=args.target_success,
            log_dir=args.output_dir,
        )
        
        # Create learner
        learner = OnlineLearner(
            policy=policy,
            env=env,
            config=config,
        )
        
        # Create dashboard for visualization
        dashboard = Dashboard(
            metrics_monitor=learner.metrics_monitor,
            output_dir=f"{args.output_dir}/plots",
        )
        
        # Run learning loop
        print("\nStarting continual learning loop...")
        print()
        
        def cycle_callback(result):
            """Called after each cycle."""
            # Update plots periodically
            if result.cycle_id % 5 == 0 and learner.cycle_history:
                try:
                    dashboard.plot_flywheel(
                        [r.__dict__ for r in learner.cycle_history],
                        save=True,
                        show=False,
                    )
                except Exception as e:
                    print(f"Warning: Could not update plot: {e}")
        
        summary = learner.run(
            num_cycles=args.cycles,
            callback=cycle_callback,
        )
        
        cycle_history = [r.__dict__ for r in learner.cycle_history]
        
        # Generate final report
        try:
            dashboard.generate_report(cycle_history, save=True)
        except Exception as e:
            print(f"Warning: Could not generate report: {e}")
        
        # Save state
        learner.save_state(f"{args.output_dir}/learner_state")
        
        env.close()
    
    # Print summary
    print()
    print("=" * 60)
    print("Continual Learning Summary")
    print("=" * 60)
    print(f"Total Cycles: {summary['total_cycles']}")
    print(f"Initial Success: {summary['initial_success_rate']:.1%}")
    print(f"Final Success: {summary['final_success_rate']:.1%}")
    print(f"Improvement: {summary['improvement']:+.1%}")
    print(f"Target Reached: {'Yes' if summary['target_reached'] else 'No'}")
    
    # Save summary
    output_path = Path(args.output_dir) / "continual_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "summary": summary,
        "cycle_history": cycle_history if 'cycle_history' in dir() else [],
        "args": vars(args),
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
