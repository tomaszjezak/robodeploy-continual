#!/usr/bin/env python3
"""
Plot the flywheel success rate climb from log files.

Usage:
    python scripts/plot_flywheel.py --log-dir=logs/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Plot flywheel success climb")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory containing cycle_history.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: {log-dir}/plots/flywheel_climb.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively",
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    
    # Try different file formats
    cycle_history = []
    
    # Try JSONL format
    jsonl_path = log_dir / "cycle_history.jsonl"
    if jsonl_path.exists():
        print(f"Loading from: {jsonl_path}")
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    cycle_history.append(json.loads(line))
    
    # Try JSON format
    if not cycle_history:
        json_path = log_dir / "continual_summary.json"
        if json_path.exists():
            print(f"Loading from: {json_path}")
            with open(json_path) as f:
                data = json.load(f)
                cycle_history = data.get("cycle_history", [])
    
    if not cycle_history:
        print(f"No cycle history found in {log_dir}")
        print("Expected files: cycle_history.jsonl or continual_summary.json")
        return
    
    print(f"Loaded {len(cycle_history)} cycles")
    
    # Import and create dashboard
    from src.continual.dashboard import Dashboard
    
    output_dir = args.output or str(log_dir / "plots")
    dashboard = Dashboard(output_dir=output_dir)
    
    # Plot
    fig = dashboard.plot_flywheel(
        cycle_history,
        save=True,
        show=args.show,
    )
    
    output_path = Path(output_dir) / "flywheel_climb.png"
    print(f"Saved plot to: {output_path}")
    
    # Print summary
    if cycle_history:
        initial = cycle_history[0].get("success_rate_before", 0)
        final = cycle_history[-1].get("success_rate_after", 0)
        print(f"\nSuccess Rate: {initial:.1%} → {final:.1%} ({final - initial:+.1%})")


if __name__ == "__main__":
    main()
