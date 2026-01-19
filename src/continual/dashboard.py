"""
Metrics Dashboard for visualizing continual learning progress.

Provides:
- Success rate over time (flywheel climb)
- Failure category distribution
- Entropy trends
- Update trigger history
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from datetime import datetime

from .metrics_monitor import MetricsMonitor, ReliabilityMetrics


class Dashboard:
    """
    Dashboard for visualizing continual learning metrics.
    
    Creates plots for:
    - Flywheel climb (success rate improvement)
    - Failure distribution
    - Entropy trends
    - Learning curves
    """
    
    def __init__(
        self,
        metrics_monitor: Optional[MetricsMonitor] = None,
        output_dir: str = "logs/plots",
    ):
        """
        Initialize dashboard.
        
        Args:
            metrics_monitor: Metrics monitor to visualize
            output_dir: Directory for saving plots
        """
        self.metrics_monitor = metrics_monitor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Style settings
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'success': '#2ecc71',
            'failure': '#e74c3c',
            'entropy': '#3498db',
            'threshold': '#e67e22',
            'update': '#9b59b6',
        }
    
    def plot_flywheel(
        self,
        cycle_history: List[Dict],
        save: bool = True,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot the flywheel success rate climb.
        
        Args:
            cycle_history: List of cycle results
            save: Save plot to file
            show: Display plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        cycles = [c.get('cycle_id', i) for i, c in enumerate(cycle_history)]
        success_before = [c.get('success_rate_before', 0) for c in cycle_history]
        success_after = [c.get('success_rate_after', 0) for c in cycle_history]
        
        # Plot success rates
        ax.plot(cycles, success_before, 'o-', color=self.colors['success'], 
                alpha=0.5, label='Before Update', linewidth=2)
        ax.plot(cycles, success_after, 's-', color=self.colors['success'],
                label='After Update', linewidth=2)
        
        # Mark updates
        for i, c in enumerate(cycle_history):
            if c.get('update_triggered', False):
                ax.axvline(x=cycles[i], color=self.colors['update'], 
                          alpha=0.3, linestyle='--')
        
        # Target line
        ax.axhline(y=0.90, color=self.colors['threshold'], linestyle='--',
                  label='90% Target', alpha=0.7)
        ax.axhline(y=0.95, color=self.colors['threshold'], linestyle=':',
                  label='95% Stretch', alpha=0.5)
        
        ax.set_xlabel('Cycle', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Flywheel: Success Rate Climb', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Add improvement annotation
        if success_before and success_after:
            improvement = success_after[-1] - success_before[0]
            ax.annotate(f'Total Improvement: {improvement:+.1%}',
                       xy=(0.02, 0.98), xycoords='axes fraction',
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'flywheel_climb.png', dpi=150)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_failure_distribution(
        self,
        failure_counts: Dict[str, int],
        save: bool = True,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot failure category distribution.
        
        Args:
            failure_counts: Dict of category -> count
            save: Save plot to file
            show: Display plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter out success and zero counts
        categories = [k for k, v in failure_counts.items() 
                     if k != 'success' and v > 0]
        counts = [failure_counts[k] for k in categories]
        
        if not categories:
            ax.text(0.5, 0.5, 'No failures recorded', 
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            # Create bar chart
            colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(categories)))
            bars = ax.barh(categories, counts, color=colors)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                       str(count), va='center', fontsize=10)
            
            ax.set_xlabel('Count', fontsize=12)
            ax.set_title('Failure Category Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'failure_distribution.png', dpi=150)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_entropy_trend(
        self,
        metrics_history: List[ReliabilityMetrics],
        threshold: float = 2.0,
        save: bool = True,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot entropy trend over time.
        
        Args:
            metrics_history: List of metrics snapshots
            threshold: Entropy threshold for "pre-fail"
            save: Save plot to file
            show: Display plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = [m.episode_count for m in metrics_history]
        mean_entropy = [m.mean_entropy for m in metrics_history]
        max_entropy = [m.max_entropy for m in metrics_history]
        
        ax.plot(episodes, mean_entropy, '-', color=self.colors['entropy'],
                label='Mean Entropy', linewidth=2)
        ax.fill_between(episodes, mean_entropy, max_entropy,
                       color=self.colors['entropy'], alpha=0.2,
                       label='Mean-Max Range')
        
        # Threshold line
        ax.axhline(y=threshold, color=self.colors['threshold'], linestyle='--',
                  label=f'Pre-fail Threshold ({threshold})', alpha=0.7)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Action Entropy', fontsize=12)
        ax.set_title('Action Entropy Trend', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'entropy_trend.png', dpi=150)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_learning_curve(
        self,
        training_history: List[Dict],
        save: bool = True,
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot training loss curve.
        
        Args:
            training_history: List of training step records
            save: Save plot to file
            show: Display plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = [h.get('step', i) for i, h in enumerate(training_history)]
        losses = [h.get('loss', 0) for h in training_history]
        
        ax.plot(steps, losses, '-', color='#2c3e50', linewidth=1.5, alpha=0.7)
        
        # Smoothed line
        if len(losses) > 10:
            window = min(20, len(losses) // 5)
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            smooth_steps = steps[window-1:]
            ax.plot(smooth_steps, smoothed, '-', color=self.colors['success'],
                   linewidth=2, label='Smoothed')
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'learning_curve.png', dpi=150)
        
        if show:
            plt.show()
        
        return fig
    
    def generate_report(
        self,
        cycle_history: List[Dict],
        training_history: Optional[List[Dict]] = None,
        save: bool = True,
    ) -> Dict[str, str]:
        """
        Generate all plots for a complete report.
        
        Args:
            cycle_history: Cycle results
            training_history: Training step records
            save: Save plots
            
        Returns:
            Dict mapping plot name to file path
        """
        paths = {}
        
        # Flywheel
        self.plot_flywheel(cycle_history, save=save)
        paths['flywheel'] = str(self.output_dir / 'flywheel_climb.png')
        
        # Failure distribution (from metrics monitor)
        if self.metrics_monitor:
            analysis = self.metrics_monitor.get_failure_analysis()
            if 'counts' in analysis:
                self.plot_failure_distribution(analysis['counts'], save=save)
                paths['failures'] = str(self.output_dir / 'failure_distribution.png')
            
            # Entropy
            if self.metrics_monitor._metrics_history:
                self.plot_entropy_trend(
                    self.metrics_monitor._metrics_history,
                    threshold=self.metrics_monitor.entropy_threshold,
                    save=save,
                )
                paths['entropy'] = str(self.output_dir / 'entropy_trend.png')
        
        # Learning curve
        if training_history:
            self.plot_learning_curve(training_history, save=save)
            paths['learning'] = str(self.output_dir / 'learning_curve.png')
        
        print(f"Generated {len(paths)} plots in {self.output_dir}")
        return paths


def quick_plot_flywheel(log_dir: str = "logs"):
    """
    Quick utility to plot flywheel from log files.
    
    Usage: python -c "from src.continual.dashboard import quick_plot_flywheel; quick_plot_flywheel()"
    """
    log_path = Path(log_dir) / "cycle_history.jsonl"
    
    if not log_path.exists():
        print(f"No cycle history found at {log_path}")
        return
    
    # Load cycle history
    cycles = []
    with open(log_path) as f:
        for line in f:
            cycles.append(json.loads(line))
    
    # Create dashboard and plot
    dashboard = Dashboard(output_dir=log_dir + "/plots")
    dashboard.plot_flywheel(cycles, save=True, show=True)
    print(f"Saved flywheel plot to {log_dir}/plots/flywheel_climb.png")
