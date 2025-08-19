import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import numpy as np
import time


class CustomAntWrapper(gym.Wrapper):
    def __init__(self, env, forward_weight=2.0, sideways_penalty_weight=1.0, healthy_reward=1.0):
        """
        Initializes the wrapper with weights for the reward function.

        Args:
            env: The environment to wrap.
            forward_weight (float): The coefficient for the forward velocity (x-velocity) reward.
            sideways_penalty_weight (float): The coefficient for the sideways velocity (y-velocity) penalty.
            healthy_reward (float): The constant reward for not terminating.
            initial_pos (tuple, optional): The initial (x, y, z) position for the ant.
        """
        super().__init__(env)
        self.forward_weight = forward_weight
        self.sideways_penalty_weight = sideways_penalty_weight
        self.healthy_reward = healthy_reward

    def step(self, action):
        """
        Applies the action and calculates a new reward to encourage straight forward walking.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # --- NEW REWARD LOGIC FOR STRAIGHT WALKING ---
        x_velocity = info.get('x_velocity', 0)
        y_velocity = info.get('y_velocity', 0)
        ctrl_cost = info.get('reward_ctrl', 0) # This is a negative value (penalty)

        # 1. Reward for Forward Motion: Strong incentive to move in the positive X direction.
        forward_reward = self.forward_weight * x_velocity

        # 2. Penalty for Sideways Motion: Penalize movement in the Y direction to keep it straight.
        sideways_penalty = self.sideways_penalty_weight * np.abs(y_velocity)
        
        # 3. Survival Reward & Control Cost: Keep the agent alive and efficient.
        
        # Combine the components to get the final reward
        modified_reward = forward_reward + self.healthy_reward + ctrl_cost - sideways_penalty
        
        return obs, modified_reward, terminated, truncated, info

# --- 1. Setup ---
# Create directories to save logs and models
log_dir = "/tmp/gym/"
model_dir = "models/PPO"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Define the model filename
model_path = os.path.join(model_dir, "ppo_ant_straight")

wrapper_kwargs = {
        'forward_weight': 1.5,
        'sideways_penalty_weight': 3,
        'healthy_reward': 5
    }

# Set to True to train a new model, False to load and view an existing one
TRAIN_MODEL = False

if TRAIN_MODEL:
    # --- 2. Training ---
    print("Creating environment and starting training...")

    # Create the vectorized environment with the custom wrapper and its arguments.
    env = make_vec_env(
        'Ant-v5',
        n_envs=4,
        wrapper_class=CustomAntWrapper,
        wrapper_kwargs=wrapper_kwargs
    )

    # Instantiate the PPO agent
    model = PPO(
        'MlpPolicy',
        env,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log=log_dir
    )

    # Train the agent for more timesteps to learn the new objective
    total_timesteps = 1000000
    print(f"Training model for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    # Save the trained model
    print(f"Training complete. Saving model to {model_path}")
    model.save(model_path)
    print("Model saved.")

    # Close the environment
    env.close()

# --- 3. Evaluation and Visualization ---
print("\n--- Loading and Evaluating Model ---")

# Check if the model file exists
if not os.path.exists(model_path + ".zip"):
    print(f"Error: Model not found at {model_path}.zip")
    print("Please run the script with TRAIN_MODEL = True first.")
else:
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    print("Model loaded.")

    base_env = gym.make('Ant-v5', render_mode='human',width=1280, height=720)
    eval_env = CustomAntWrapper(base_env, **wrapper_kwargs)

    # Reset the environment to get the initial observation
    obs, info = eval_env.reset()

    print("Starting evaluation. The simulation window will open.")
    # Run the simulation loop
    try:
        for _ in range(1000): # Run for 5000 steps
            # Get the action from the model
            action, _states = model.predict(obs, deterministic=True)

            # Take the action in the environment
            obs, reward, terminated, truncated, info = eval_env.step(action)

            # If the episode is over, reset the environment
            if terminated or truncated:
                print("Episode finished. Resetting environment.")
                obs, info = eval_env.reset()

            time. sleep(1/30)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        # Close the environment window
        eval_env.close()
        print("Evaluation finished and environment closed.")
