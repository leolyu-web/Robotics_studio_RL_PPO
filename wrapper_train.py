import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import myant_env 

gymnasium.register(
     id='MyAnt-v0',
     entry_point='myant_env:myAntEnv' # The format is 'filename:ClassName'
)

# --- Define the Circle Reward Wrapper ---
class CircleRewardWrapper(gymnasium.Wrapper):
    """
    A wrapper that modifies the reward to encourage circular motion.
    """
    def __init__(self, env, yaw_reward_weight=0.5):

        super().__init__(env)
        self.yaw_reward_weight = yaw_reward_weight

    def step(self, action):
        """
        Overrides the step method to modify the reward.
        """
        # Call the original environment's step function
        observation, reward, terminated, truncated, info = self.env.step(action)

        # --- Recalculate the total reward ---
        # Get the original reward components from the info dict
        healthy_reward = info['reward_healthy']
        forward_reward = info['reward_forward'] # This is (world x_velocity * weight)
        ctrl_cost = -info['cost_ctrl']
        contact_cost = -info['cost_contact']

        # Get the yaw velocity (angular velocity around z-axis)
        yaw_velocity = self.env.unwrapped.data.qvel[5]

        # Calculate the new yaw reward
        yaw_reward = self.yaw_reward_weight * yaw_velocity

        # Calculate the new total reward
        new_total_reward = (
            healthy_reward
            + forward_reward
            + yaw_reward
            - ctrl_cost
            - contact_cost
        )

        # --- Update the info dictionary ---
        info['reward_yaw'] = yaw_reward
        info['reward_total'] = new_total_reward # Overwrite the original total

        # Return the modified reward
        return observation, new_total_reward, terminated, truncated, info
    # No need to override reset, the default Wrapper.reset()
    
if __name__ == '__main__':
    env_id = 'MyAnt-v0'
    
    num_cpu = 4
    
    wrapper_kwargs = {'yaw_reward_weight': 0.7} 

    print("Creating vectorized environment with CircleRewardWrapper...")
    env = make_vec_env(
        env_id, 
        n_envs=num_cpu,
        wrapper_class=CircleRewardWrapper,   # <Tell make_vec_env to use our wrapper
        wrapper_kwargs=wrapper_kwargs        # <Pass arguments to the wrapper's __init__
    )
    
    # --- 3. Set up the PPO Model ---
    model = PPO(
        'MlpPolicy',        # The policy network architecture
        env,                # The environment to train on
        n_steps=1024,       # Steps to collect per environment before update
        batch_size=64,      # Mini-batch size for gradient descent
        n_epochs=10,        # Number of epochs (passes) over the data per update
        gamma=0.99,         # Discount factor for future rewards
        gae_lambda=0.95,    # Factor for Generalized Advantage Estimation (GAE)
        clip_range=0.2,     # Clipping parameter (the "P" in PPO)
        ent_coef=0.0,       # Entropy coefficient (for exploration)
        verbose=1,          # Verbosity level (0=none, 1=logs)
        tensorboard_log=None # Directory for TensorBoard logs
    )
    
    print("Starting model training...")
    model.learn(total_timesteps=1000000)
    
    print("Training finished. Saving model...")
    model.save("PPO_model/ppo_myant_circle_model")
    
    env.close()

    print("Model saved as ppo_myant_circle_model.zip")