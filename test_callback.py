#!/usr/bin/env python3
"""
Test script to check if callback is working properly
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class TestCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.call_count = 0

    def _on_step(self) -> bool:
        self.call_count += 1
        if self.call_count % 1000 == 0:  # Print every 1000 steps
            print(f"Callback called at step {self.n_calls}, total calls: {self.call_count}")
        return True

def test_callback():
    print("Testing callback functionality...")

    # Create simple environment
    env = gym.make('stocks-v0')
    env = gym.wrappers.RecordEpisodeStatistics(env)  # Add stats wrapper

    # Create model
    model = PPO("MlpPolicy", env, n_steps=512, batch_size=256, verbose=0)

    # Create callback
    callback = TestCallback(verbose=1)

    print("Starting training with callback...")
    # Train for short time to test
    model.learn(total_timesteps=5000, callback=callback, progress_bar=True)

    print(f"Final callback call count: {callback.call_count}")
    print(f"Training completed at {model.num_timesteps} timesteps")

    env.close()

if __name__ == "__main__":
    test_callback()
