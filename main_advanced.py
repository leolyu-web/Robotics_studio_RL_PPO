import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation
from env.env_func import create_uneven_terrain


class CustomAntWrapper(gym.Wrapper):
    def __init__(self, env, forward_weight=2.0, sideways_penalty_weight=1.0, healthy_reward=1.0, initial_pos=None, healthy_z_range=(0.2, 1.0), orientation_threshold=0.0):
        """
        Initializes the wrapper with weights for the reward function and custom termination conditions.

        Args:
            env: The environment to wrap.
            forward_weight (float): The coefficient for the forward velocity (x-velocity) reward.
            sideways_penalty_weight (float): The coefficient for the sideways velocity (y-velocity) penalty.
            healthy_reward (float): The constant reward for not terminating.
            initial_pos (tuple, optional): The initial (x, y, z) position for the ant.
            healthy_z_range (tuple): The (min, max) Z-height for the ant's torso to be considered healthy.
            orientation_threshold (float): The minimum value for the z-component of the
                                           torso's 'up' vector to be considered healthy.
        """
        super().__init__(env)
        # Reward parameters
        self.forward_weight = forward_weight
        self.sideways_penalty_weight = sideways_penalty_weight
        self.healthy_reward = healthy_reward
        
        # Initial position parameters
        self.initial_pos = None
        if initial_pos is not None:
            self.initial_pos = np.array(initial_pos)
            assert hasattr(self.env.unwrapped, "set_state"), "Environment must have a 'set_state' method to set position."
            assert hasattr(self.env.unwrapped, "_get_obs"), "Environment must have a '_get_obs' method to get new observation."

        # Custom termination parameters
        self._healthy_z_range = healthy_z_range
        self._orientation_threshold = orientation_threshold

    def is_healthy(self):
        """
        Checks if the ant is in a "healthy" state based on height and orientation.
        This logic is now part of the wrapper.
        """
        # 1. Height Check
        min_z, max_z = self._healthy_z_range
        is_within_height_range = self.env.unwrapped.data.qpos[2] >= min_z and self.env.unwrapped.data.qpos[2] <= max_z

        # 2. Orientation Check (using scipy)
        q = self.env.unwrapped.data.qpos[3:7]  # Get quaternion in [w, x, y, z] format
        
        # Reorder quaternion to [x, y, z, w] for scipy
        q_scipy = q[[1, 2, 3, 0]] 
        
        # Create a rotation object and apply it to the 'up' vector [0, 0, 1]
        rot = Rotation.from_quat(q_scipy)
        rotated_up_vector = rot.apply([0, 0, 1])
        z_up = rotated_up_vector[2] # Extract the Z-component
        
        is_oriented = z_up > self._orientation_threshold

        return is_within_height_range and is_oriented

    def step(self, action):
        """
        Applies the action, calculates a new reward, and checks for termination.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # --- NEW REWARD LOGIC FOR STRAIGHT WALKING ---
        x_velocity = info.get('x_velocity', 0)
        y_velocity = info.get('y_velocity', 0)
        ctrl_cost = info.get('reward_ctrl', 0) # This is a negative value (penalty)

        # 1. Reward for Forward Motion
        forward_reward = self.forward_weight * x_velocity

        # 2. Penalty for Sideways Motion
        sideways_penalty = self.sideways_penalty_weight * np.abs(y_velocity)
        
        # Combine the components to get the final reward
        modified_reward = forward_reward + self.healthy_reward + ctrl_cost - sideways_penalty
        
        # --- CUSTOM TERMINATION LOGIC ---
        # The episode is terminated if the ant is not healthy
        healthy = self.is_healthy()
        terminated = False
        if not healthy:
            terminated = True
            
        return obs, modified_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Resets the environment and then sets a custom initial position if one was specified.
        """
        obs, info = self.env.reset(**kwargs)

        if self.initial_pos is not None:
            qpos = self.env.unwrapped.data.qpos
            qvel = self.env.unwrapped.data.qvel
            qpos[:3] = self.initial_pos
            self.env.unwrapped.set_state(qpos, qvel)
            obs = self.env.unwrapped._get_obs()

        return obs, info


def main():
    custom_xml_path = create_uneven_terrain(max_height=1.5)

    log_dir = "/tmp/gym/"
    model_dir = "models/PPO"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "ppo_ant_advanced")

    TRAIN_MODEL = False
    start_pos = (-1, 0, 1.25)

    if TRAIN_MODEL:
        print("\nCreating custom uneven environment and starting training...")

        def make_env():
            base_env = gym.make('Ant-v5', xml_file=custom_xml_path)
            
            # Pass all custom parameters to the wrapper
            wrapped_env = CustomAntWrapper(base_env, 
                                           forward_weight=1.2,    
                                           sideways_penalty_weight=0.2,
                                           healthy_reward=1.0,
                                           initial_pos=start_pos,
                                           healthy_z_range=(-3.0, 3.0),
                                           orientation_threshold=-0.5)
            return wrapped_env

        env = make_vec_env(make_env, n_envs=4)

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

        total_timesteps = 2000000
        print(f"Training model for {total_timesteps} timesteps on uneven terrain...")
        model.learn(total_timesteps=total_timesteps)

        print(f"Training complete. Saving model to {model_path}")
        model.save(model_path)
        print("Model saved.")

        env.close()

    print("\n--- Loading and Evaluating Model on Uneven Terrain ---")

    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model not found at {model_path}.zip")
        print("Please run the script with TRAIN_MODEL = True first.")
    else:
        print(f"Loading model from {model_path}...")
        model = PPO.load(model_path)
        print("Model loaded.")

        print("Creating evaluation environment.")
        # Create the base Ant environment
        base_env = gym.make("Ant-v5", 
                            xml_file=custom_xml_path, 
                            render_mode='human',
                            width=1280, height=720)

        # Apply the wrapper with all customizations
        eval_env = CustomAntWrapper(base_env, 
                                   forward_weight=0.4,    
                                   sideways_penalty_weight=0.1,
                                   healthy_reward=1.0,
                                   initial_pos=start_pos,
                                   healthy_z_range=(-3.0, 3.0),
                                   orientation_threshold=-0.5)

        obs, info = eval_env.reset()

        print("Starting evaluation. The simulation window will open.")
        try:
            for _ in range(10000):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                time.sleep(1/30)

                if terminated or truncated:
                    print("Episode finished. Resetting environment.")
                    obs, info = eval_env.reset()

        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user.")
        finally:
            eval_env.close()
            print("Evaluation finished and environment closed.")

if __name__ == "__main__" :
    main()
