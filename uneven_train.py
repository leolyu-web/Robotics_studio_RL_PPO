import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import myant_env  
import numpy as np
import mujoco

gymnasium.register(
     id='MyAnt-v0',
     entry_point='myant_env:myAntEnv', # The format is 'filename:ClassName'
     max_episode_steps=1000
)

class uneven_Wrapper(gymnasium.Wrapper):
    """
    This wrapper modifies the MyAnt-v0 environment to:
    1.  Randomize the initial (z) position at reset.
    2.  Add an orientation check to the 'is_healthy' condition.
    3.  Override the base environment's 'healthy_z_range'.
    """
    
    def __init__(self, 
                 env, 
                 #  Fixed default value to be a tuple, not a list of tuples
                 reset_z_range=(0.75, 1.0), 
                 healthy_orientation_threshold=0.2,
                 # --- Add parameter for new z-range ---
                 new_healthy_z_range=None):
        """
        Args:
            env: The environment to wrap.
            reset_z_range: A tuple (z_min, z_max) for randomizing
                           the start height.
            healthy_orientation_threshold: The minimum acceptable z-component 
                                           of the torso's 'up' vector.
            new_healthy_z_range: (Optional) A tuple (z_min, z_max) to
                                 override the base env's healthy_z_range.
        """
        super().__init__(env)
        
        self.reset_z_range = reset_z_range
        self.healthy_orientation_threshold = healthy_orientation_threshold
        
        # Pre-allocate memory for the rotation matrix
        self._rot_mat = np.zeros(9)

        # --- OVERRIDE THE BASE ENV'S HEALTHY Z RANGE ---
        if new_healthy_z_range is not None:
            # .unwrapped accesses the original myAntEnv instance
            self.env.unwrapped.healthy_z_range = new_healthy_z_range

    def _is_upright(self):
        """
        Checks if the ant's torso is upright.
        """
        # Using .unwrapped is safer when accessing base env attributes
        quat = self.env.unwrapped.data.qpos[3:7] 
        mujoco.mju_quat2Mat(self._rot_mat, quat)
        up_z = self._rot_mat[8]
        return up_z > self.healthy_orientation_threshold

    def step(self, action):
        """
        Wraps the 'step' method to add the orientation health check.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        is_upright = self._is_upright()
        
        new_terminated = terminated or (not is_upright)
        
        if new_terminated and not terminated:
            healthy_reward_param = self.env.unwrapped.healthy_reward
            reward -= healthy_reward_param
            
            if 'reward_total' in info:
                info['reward_total'] -= healthy_reward_param
            if 'reward_healthy' in info:
                info['reward_healthy'] = 0.0
            
        return obs, reward, new_terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """
        Wraps the 'reset' method to randomize the initial z position.
        """
        super().reset(seed=seed) 
        
        obs, info = self.env.reset(seed=seed, options=options)
        
        # --- Override the (z) position ---
        
        low_z, high_z = self.reset_z_range
        new_z = self.np_random.uniform(low=low_z, high=high_z)
        
        # Using .unwrapped is safer when accessing base env attributes
        qpos = self.env.unwrapped.data.qpos.copy()
        
        qpos[2] = new_z
        
        self.env.unwrapped.data.qpos[:] = qpos
        
        mujoco.mj_forward(self.env.unwrapped.model, self.env.unwrapped.data)
        
        obs = self.env.unwrapped._get_obs()
        
        info['x_position'] = qpos[0]
        info['y_position'] = qpos[1]
        info['distance_from_origin'] = np.linalg.norm(qpos[0:2], ord=2)
        
        return obs, info


wrapper_kwargs = dict(
        # Set the parameter
        reset_z_range=(2.55 , 2.56),  
        healthy_orientation_threshold=0.2,
        new_healthy_z_range=(-5.0, 5.0)
    )
    
if __name__ == '__main__':
    env_id = 'MyAnt-v0'
    
    # --- Create Vectorized Environment ---
    num_cpu = 4
    
    env = make_vec_env(
        env_id, 
        n_envs=num_cpu,
        wrapper_class=uneven_Wrapper,
        wrapper_kwargs=wrapper_kwargs,
        env_kwargs={'xml_path': "assets/ant_uneven.xml"}
    )
    
    # --- Set up the PPO Model ---
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
        tensorboard_log=None
    )
    
    # --- Train the Model ---
    print("Starting model training with custom wrapper...")
    model.learn(total_timesteps=1000000)
    
    # --- Save the Model ---
    print("Training finished. Saving model...")
    model.save("PPO_model/ppo_myant_uneven_model")
    
    env.close()