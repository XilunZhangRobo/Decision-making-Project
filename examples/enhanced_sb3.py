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

import gym_anytrading
from gym_anytrading import datasets


SEED = 42


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


def train_agent(env_id: str, timesteps: int, learning_rate: float = 3e-4, n_envs: int = 4):
    vec_env = SubprocVecEnv([make_env(env_id) for _ in range(n_envs)])

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

    model.learn(total_timesteps=timesteps, progress_bar=True)
    vec_env.close()
    return model


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


def main():
    # keep a copy of dataset so repeated runs are deterministic
    gym_anytrading.datasets.STOCKS_GOOGL = deepcopy(datasets.STOCKS_GOOGL)

    print("=== Baseline env (stocks-v0) ===")
    baseline_random = random_baseline("stocks-v0", episodes=5)
    print(f"Random baseline profit: {baseline_random}")

    baseline_model = train_agent("stocks-v0", timesteps=500_000, n_envs=4)
    base_env = gym.make("stocks-v0")
    baseline_eval = evaluate(baseline_model, base_env, episodes=5)
    print(f"PPO baseline profit:    {baseline_eval}")

    print("\n=== Enhanced env (stocks-enhanced-v0) - Risk-Adjusted Focus ===")
    enhanced_random = random_baseline("stocks-enhanced-v0", episodes=5)
    print(f"Random baseline profit: {enhanced_random}")

    # Try enhanced with different approach: better reward function
    enhanced_model_ppo = train_agent("stocks-enhanced-v0", timesteps=200_000, learning_rate=3e-4, n_envs=4)
    enhanced_env = gym.make("stocks-enhanced-v0")
    enhanced_eval_ppo = evaluate(enhanced_model_ppo, enhanced_env, episodes=5)
    print(f"PPO enhanced profit:    {enhanced_eval_ppo}")

    # Compare improvement
    baseline_profit = baseline_eval["mean"]
    enhanced_profit = enhanced_eval_ppo["mean"]
    improvement = ((enhanced_profit - baseline_profit) / baseline_profit) * 100
    print(".1f")

    # Note: TD3 removed since it requires continuous actions, but we now use discrete actions like baseline


if __name__ == "__main__":
    main()

