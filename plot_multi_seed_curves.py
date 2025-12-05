#!/usr/bin/env python3
"""
Plot the multi-seed curves if data exists, otherwise plot single seed with note.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def plot_available_data():
    """Plot whatever data we have available."""

    # Try to load multi-seed data
    try:
        with open('examples/multi_seed_training_curves.json', 'r') as f:
            data = json.load(f)

        if len(data.get('baseline', [])) > 1:
            print("Found multi-seed data! Plotting with variance...")
            plot_multi_seed(data)
        else:
            print("Only single seed data found. Plotting single curves...")
            plot_single_seed_comparison()

    except FileNotFoundError:
        print("No multi-seed data found. Plotting single curves...")
        plot_single_seed_comparison()

def plot_single_seed_comparison():
    """Plot the basic comparison from single runs."""
    try:
        with open('training_curves.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("No training data found!")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Extract data
    baseline_timesteps = [d['timestep'] for d in data['baseline']]
    baseline_profits = [d['mean_profit'] for d in data['baseline']]
    enhanced_timesteps = [d['timestep'] for d in data['enhanced']]
    enhanced_profits = [d['mean_profit'] for d in data['enhanced']]

    # Plot 1: Baseline PPO
    ax1.plot(baseline_timesteps, baseline_profits, 'b-', linewidth=3, marker='o', markersize=6, label='Baseline PPO')
    ax1.axhline(y=baseline_profits[-1], color='darkblue', linestyle='--', alpha=0.7, label='Final Performance')
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Mean Profit', fontsize=12)
    ax1.set_title('Baseline PPO: Stuck at Local Optimum', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_ylim([0, max(baseline_profits) * 1.2])

    # Plot 2: Enhanced PPO
    ax2.plot(enhanced_timesteps, enhanced_profits, 'r-', linewidth=3, marker='s', markersize=6, label='Enhanced PPO')
    ax2.axhline(y=enhanced_profits[-1], color='darkred', linestyle='--', alpha=0.7, label='Final Performance')
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Mean Profit', fontsize=12)
    ax2.set_title('Enhanced PPO: Continuous Learning', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_ylim([0, max(enhanced_profits) * 1.2])

    # Plot 3: Comparison
    ax3.plot(baseline_timesteps, baseline_profits, 'b-', linewidth=3, marker='o', markersize=6, label='Baseline PPO')
    ax3.plot(enhanced_timesteps, enhanced_profits, 'r-', linewidth=3, marker='s', markersize=6, label='Enhanced PPO')
    ax3.axhline(y=baseline_profits[-1], color='blue', linestyle='--', alpha=0.5, label='Baseline Final')
    ax3.axhline(y=enhanced_profits[-1], color='red', linestyle='--', alpha=0.5, label='Enhanced Final')
    ax3.set_xlabel('Training Steps', fontsize=12)
    ax3.set_ylabel('Mean Profit', fontsize=12)
    ax3.set_title('Direct Comparison', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)

    # Plot 4: Performance improvement
    improvement = [(e - b) / b * 100 for e, b in zip(enhanced_profits, baseline_profits[:len(enhanced_profits)])]
    ax4.plot(enhanced_timesteps, improvement, 'g-', linewidth=3, marker='^', markersize=6, label='Performance Improvement')
    final_improvement = (enhanced_profits[-1] - baseline_profits[-1]) / baseline_profits[-1] * 100
    ax4.axhline(y=final_improvement, color='purple', linestyle='--', alpha=0.7, linewidth=2, label=f'Final: {final_improvement:.1f}% Better')
    ax4.set_xlabel('Training Steps', fontsize=12)
    ax4.set_ylabel('Performance Improvement (%)', fontsize=12)
    ax4.set_title('Relative Performance Gain', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('ppo_comparison_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Plots saved as 'ppo_comparison_curves.png'")
    print(".3f")
    print(".3f")
    print(".1f")

def plot_multi_seed(data):
    """Plot multi-seed data with variance bands."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Extract data
    random_baseline = data['random_baseline']['mean']
    random_enhanced = data['random_enhanced']['mean']

    # Process baseline data across seeds
    baseline_curves = []
    baseline_finals = []
    for result in data['baseline']:
        curve = result['training_curve']
        baseline_curves.append([d['mean_profit'] for d in curve])
        baseline_finals.append(result['final_eval']['mean'])

    # Process enhanced data across seeds
    enhanced_curves = []
    enhanced_finals = []
    for result in data['enhanced']:
        curve = result['training_curve']
        enhanced_curves.append([d['mean_profit'] for d in curve])
        enhanced_finals.append(result['final_eval']['mean'])

    # Convert to numpy arrays for easier plotting
    baseline_curves = np.array(baseline_curves)
    enhanced_curves = np.array(enhanced_curves)
    timesteps = [d['timestep'] for d in data['baseline'][0]['training_curve']]

    # Calculate means and stds
    baseline_mean = np.mean(baseline_curves, axis=0)
    baseline_std = np.std(baseline_curves, axis=0)
    enhanced_mean = np.mean(enhanced_curves, axis=0)
    enhanced_std = np.std(enhanced_curves, axis=0)

    # Plot 1: Baseline PPO with variance
    ax1.plot(timesteps, baseline_mean, 'b-', linewidth=3, label='Baseline PPO (Mean)')
    ax1.fill_between(timesteps, baseline_mean - baseline_std, baseline_mean + baseline_std,
                     color='blue', alpha=0.2)
    ax1.axhline(y=random_baseline, color='gray', linestyle='--', linewidth=2,
                label=f'Random: {random_baseline:.4f}')
    ax1.axhline(y=np.mean(baseline_finals), color='darkblue', linestyle='-', linewidth=2,
                label=f'Final: {np.mean(baseline_finals):.3f}')
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Mean Profit', fontsize=12)
    ax1.set_title('Baseline PPO: Stuck at Local Optimum', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, max(baseline_mean + baseline_std) * 1.2])

    # Plot 2: Enhanced PPO with variance
    ax2.plot(timesteps, enhanced_mean, 'r-', linewidth=3, label='Enhanced PPO (Mean)')
    ax2.fill_between(timesteps, enhanced_mean - enhanced_std, enhanced_mean + enhanced_std,
                     color='red', alpha=0.2)
    ax2.axhline(y=random_enhanced, color='gray', linestyle='--', linewidth=2,
                label=f'Random: {random_enhanced:.4f}')
    ax2.axhline(y=np.mean(enhanced_finals), color='darkred', linestyle='-', linewidth=2,
                label=f'Final: {np.mean(enhanced_finals):.3f}')
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Mean Profit', fontsize=12)
    ax2.set_title('Enhanced PPO: Continuous Learning', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, max(enhanced_mean + enhanced_std) * 1.2])

    # Plot 3: Direct comparison with variance
    ax3.plot(timesteps, baseline_mean, 'b-', linewidth=3, label='Baseline PPO')
    ax3.fill_between(timesteps, baseline_mean - baseline_std, baseline_mean + baseline_std,
                     color='blue', alpha=0.2)
    ax3.plot(timesteps, enhanced_mean, 'r-', linewidth=3, label='Enhanced PPO')
    ax3.fill_between(timesteps, enhanced_mean - enhanced_std, enhanced_mean + enhanced_std,
                     color='red', alpha=0.2)
    ax3.axhline(y=random_baseline, color='gray', linestyle='--', linewidth=1,
                label=f'Baseline Random: {random_baseline:.4f}')
    ax3.axhline(y=random_enhanced, color='black', linestyle='--', linewidth=1,
                label=f'Enhanced Random: {random_enhanced:.4f}')
    ax3.set_xlabel('Training Steps', fontsize=12)
    ax3.set_ylabel('Mean Profit', fontsize=12)
    ax3.set_title('Direct Comparison with Variance', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_ylim([0, max(enhanced_mean + enhanced_std) * 1.2])

    # Plot 4: Performance improvement
    improvement_mean = (enhanced_mean - baseline_mean) / baseline_mean * 100
    improvement_std = np.sqrt((enhanced_std/enhanced_mean)**2 + (baseline_std/baseline_mean)**2) * abs(improvement_mean)
    ax4.plot(timesteps, improvement_mean, 'g-', linewidth=3, label='Mean Improvement')
    ax4.fill_between(timesteps, improvement_mean - improvement_std, improvement_mean + improvement_std,
                     color='green', alpha=0.2)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_xlabel('Training Steps', fontsize=12)
    ax4.set_ylabel('Performance Improvement (%)', fontsize=12)
    ax4.set_title('Relative Performance Gain', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('ppo_multi_seed_curves_final.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\n=== MULTI-SEED EXPERIMENT SUMMARY ===")
    print(f"Random Baseline Profit: {random_baseline:.4f}")
    print(f"Random Enhanced Profit: {random_enhanced:.4f}")
    print(f"Baseline PPO Final (Mean ± Std): {np.mean(baseline_finals):.3f} ± {np.std(baseline_finals):.3f}")
    print(f"Enhanced PPO Final (Mean ± Std): {np.mean(enhanced_finals):.3f} ± {np.std(enhanced_finals):.3f}")
    print(f"Final Improvement: {((np.mean(enhanced_finals) - np.mean(baseline_finals)) / np.mean(baseline_finals) * 100):.1f}%")
    print("\nPlots saved as 'ppo_multi_seed_curves_final.png'")

if __name__ == "__main__":
    plot_available_data()
