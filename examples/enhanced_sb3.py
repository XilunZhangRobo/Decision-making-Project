"""
Train and evaluate an improved PPO trading agent on the enhanced observation/
reward design. Prints mean/min/max profit over multiple evaluation episodes.
"""

import random
from copy import deepcopy

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import plot_results

import gym_anytrading
from gym_anytrading import datasets


SEED = 42
NUM_SEEDS = 3  # Run experiments with multiple seeds for variance


class TrainingEvalCallback(BaseCallback):
    """
    Callback for evaluating agent during training and storing results for plotting.
    """
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=3, verbose=0, seed_offset=1000):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.seed_offset = seed_offset
        self.eval_results = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current policy
            profits = []
            for ep in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset(seed=SEED + self.seed_offset + ep)  # Different seed per episode
                terminated = False
                truncated = False
                total_profit = info.get("total_profit", 1.0)
                while not (terminated or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    total_profit = info.get("total_profit", total_profit)
                profits.append(total_profit)

            mean_profit = float(np.mean(profits))
            self.eval_results.append({
                'timestep': self.n_calls,
                'mean_profit': mean_profit,
                'min_profit': float(np.min(profits)),
                'max_profit': float(np.max(profits))
            })

            if self.verbose > 0:
                print(f"Timestep {self.n_calls}: Eval profit = {mean_profit:.3f}")

        return True

    def get_results(self):
        return self.eval_results


def make_env(env_id: str):
    return lambda: Monitor(gym.make(env_id))


def evaluate(model, env, episodes: int = 5):
    profits = []
    for ep in range(episodes):
        obs, info = env.reset(seed=SEED + ep)
        terminated = False
        truncated = False
        total_profit = info.get("total_profit", 1.0)
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_profit = info.get("total_profit", total_profit)
        profits.append(total_profit)
    return {
        "mean": float(np.mean(profits)),
        "min": float(np.min(profits)),
        "max": float(np.max(profits)),
    }


def train_agent(env_id: str, timesteps: int, learning_rate: float = 3e-4, n_envs: int = 4, eval_callback=None):
    vec_env = SubprocVecEnv([make_env(env_id) for _ in range(n_envs)])

    # Use provided callback or create default one
    if eval_callback is None:
        eval_env = gym.make(env_id)
        eval_callback = TrainingEvalCallback(eval_env, eval_freq=5000, n_eval_episodes=3, verbose=1, seed_offset=SEED)
        should_close_eval_env = True
    else:
        should_close_eval_env = False

    # Tune hyperparameters for enhanced observation space with fundamental features
    if "enhanced" in env_id:
        # Network for 4 simple features
        policy_kwargs = dict(net_arch=[128, 64])
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=2048,  # Reasonable rollout buffer
            batch_size=512,  # Moderate batch size
            gamma=0.995,  # Standard horizon
            gae_lambda=0.95,
            ent_coef=0.015,  # Moderate entropy
            learning_rate=learning_rate * 1.5,  # Moderate learning rate increase
            clip_range=0.2,  # PPO clip range
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=SEED,
        )
    else:
        # Baseline parameters
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=512,
            batch_size=256,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.01,
            learning_rate=learning_rate,
            verbose=0,
            seed=SEED,
        )

    model.learn(total_timesteps=timesteps, progress_bar=True, callback=eval_callback)
    vec_env.close()
    if should_close_eval_env:
        eval_env.close()

    return model, eval_callback.get_results()


def train_td3(env_id: str, timesteps: int, learning_rate: float = 3e-4, n_envs: int = 4):
    vec_env = SubprocVecEnv([make_env(env_id) for _ in range(n_envs)])
    model = TD3(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.995,
        policy_delay=2,
        train_freq=(1, "step"),
        gradient_steps=1,
        verbose=0,
        seed=SEED,
    )
    model.learn(total_timesteps=timesteps, progress_bar=True)
    vec_env.close()
    return model


def random_baseline(env_id: str, episodes: int = 5):
    print(f"Running random baseline for {env_id}...")
    env = gym.make(env_id)
    profits = []
    for ep in range(episodes):
        obs, info = env.reset(seed=SEED + ep)
        terminated = False
        truncated = False
        total_profit = info.get("total_profit", 1.0)
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_profit = info.get("total_profit", total_profit)
        profits.append(total_profit)
    env.close()
    return {
        "mean": float(np.mean(profits)),
        "min": float(np.min(profits)),
        "max": float(np.max(profits)),
    }


def create_multi_seed_plots(all_results):
    """Create learning curve plots with variance across seeds."""
    import matplotlib.pyplot as plt

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Extract data
    random_baseline = all_results['random_baseline']['mean']
    random_enhanced = all_results['random_enhanced']['mean']

    # Process baseline data across seeds
    baseline_curves = []
    baseline_finals = []
    for result in all_results['baseline']:
        curve = result['training_curve']
        baseline_curves.append([d['mean_profit'] for d in curve])
        baseline_finals.append(result['final_eval']['mean'])

    # Process enhanced data across seeds
    enhanced_curves = []
    enhanced_finals = []
    for result in all_results['enhanced']:
        curve = result['training_curve']
        enhanced_curves.append([d['mean_profit'] for d in curve])
        enhanced_finals.append(result['final_eval']['mean'])

    # Convert to numpy arrays for easier plotting
    baseline_curves = np.array(baseline_curves)
    enhanced_curves = np.array(enhanced_curves)
    timesteps = [d['timestep'] for d in all_results['baseline'][0]['training_curve']]

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
    plt.savefig('ppo_multi_seed_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\n=== MULTI-SEED EXPERIMENT SUMMARY ===")
    print(f"Random Baseline Profit: {random_baseline:.4f}")
    print(f"Random Enhanced Profit: {random_enhanced:.4f}")
    print(f"Baseline PPO Final (Mean ± Std): {np.mean(baseline_finals):.3f} ± {np.std(baseline_finals):.3f}")
    print(f"Enhanced PPO Final (Mean ± Std): {np.mean(enhanced_finals):.3f} ± {np.std(enhanced_finals):.3f}")
    print(f"Final Improvement: {((np.mean(enhanced_finals) - np.mean(baseline_finals)) / np.mean(baseline_finals) * 100):.1f}%")
    print("\nPlots saved as 'ppo_multi_seed_curves.png'")


def run_single_experiment(env_id, seed, timesteps=200000):
    """Run a single experiment with given seed."""
    # Set the seed for this experiment
    import random
    random.seed(seed)
    np.random.seed(seed)

    # Temporarily override the global SEED
    global SEED
    original_seed = SEED
    SEED = seed

    try:
        print(f"\n--- Running {env_id} with seed {seed} ---")

        # Create evaluation environment and callback
        eval_env = gym.make(env_id)
        eval_callback = TrainingEvalCallback(eval_env, eval_freq=5000, n_eval_episodes=3, verbose=1, seed_offset=seed)

        # Train agent with the callback
        model, training_curve = train_agent(env_id, timesteps=timesteps, n_envs=4, eval_callback=eval_callback)

        # Final evaluation
        final_env = gym.make(env_id)
        final_eval = evaluate(model, final_env, episodes=5)
        final_env.close()

        # Close the evaluation environment we created
        eval_env.close()

        return {
            'seed': seed,
            'training_curve': training_curve,
            'final_eval': final_eval
        }

    finally:
        # Restore original seed
        SEED = original_seed


def main():
    # keep a copy of dataset so repeated runs are deterministic
    gym_anytrading.datasets.STOCKS_GOOGL = deepcopy(datasets.STOCKS_GOOGL)

    # Store results for all seeds
    all_results = {
        'baseline': [],
        'enhanced': [],
        'random_baseline': None,
        'random_enhanced': None
    }

    print("=== Running Multiple Seeds for Variance Analysis ===")

    # Get random baselines (these are deterministic across seeds)
    print("\n=== Random Baselines ===")
    baseline_random = random_baseline("stocks-v0", episodes=5)
    enhanced_random = random_baseline("stocks-enhanced-v0", episodes=5)
    all_results['random_baseline'] = baseline_random
    all_results['random_enhanced'] = enhanced_random

    print(f"Random baseline (stocks-v0): {baseline_random}")
    print(f"Random enhanced (stocks-enhanced-v0): {enhanced_random}")

    # Run experiments with multiple seeds
    seeds = [41, 42, 43, 44, 45]  # Use different seeds for variance

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT SEED {seed}")
        print(f"{'='*60}")

        # Run baseline experiment
        baseline_result = run_single_experiment("stocks-v0", seed, timesteps=500000)  # Shorter for speed
        all_results['baseline'].append(baseline_result)

        # Run enhanced experiment
        enhanced_result = run_single_experiment("stocks-enhanced-v0", seed, timesteps=500000)
        all_results['enhanced'].append(enhanced_result)

    # Save all results
    import json
    with open("multi_seed_training_curves.json", "w") as f:
        # Convert to serializable format
        serializable_results = {
            'random_baseline': all_results['random_baseline'],
            'random_enhanced': all_results['random_enhanced'],
            'baseline': [{
                'seed': r['seed'],
                'training_curve': r['training_curve'],
                'final_eval': r['final_eval']
            } for r in all_results['baseline']],
            'enhanced': [{
                'seed': r['seed'],
                'training_curve': r['training_curve'],
                'final_eval': r['final_eval']
            } for r in all_results['enhanced']]
        }
        json.dump(serializable_results, f, indent=2)

    print("\nMulti-seed results saved to 'multi_seed_training_curves.json'")

    # Create plots with variance
    create_multi_seed_plots(all_results)

    # Note: TD3 removed since it requires continuous actions, but we now use discrete actions like baseline


if __name__ == "__main__":
    main()

