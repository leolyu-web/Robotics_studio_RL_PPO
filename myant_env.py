# myant_env.py
import os
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium
from gymnasium.spaces import Box

class myAntEnv(gymnasium.Env):
    """
    myAnt robot env
    and documentation.
    """
    metadata = {
        "render_modes": ["human"],
        "render_fps": 30
    }

    def __init__(self, render_mode=None, xml_path="assets/ant.xml"):
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found at {xml_path}")
            
        # Load the model and data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.render_mode = render_mode
        self.viewer = None
        
        # --- Define Spaces ---
        # Action space: 8 actuators (from XML) with ctrlrange -1.0 to 1.0
        self.action_space = Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        
        # Observation space:
        # qpos[2:] (z-height + 4-dim quaternion + 8 joint angles) = 1 + 4 + 8 = 13
        # qvel (3-dim linear vel + 3-dim angular vel + 8 joint vels) = 3 + 3 + 8 = 14
        # cfrc_ext (13 geoms * 6-dim force/torque vector) = 78 
        # Total obs dim = 13 + 14 + 78 = 105
        obs_shape = 105 
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        # --- Initial State ---
        # This init_qpos from the official env is 15-dim
        self.init_qpos = self.data.qpos.copy() # Get default from model
        self.init_qpos[0:7] = [0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0]
        # self.init_qpos[7:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0] # Your values
        self.init_qpos[7:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Your values
        
        self.init_qvel = np.zeros(self.model.nv) # 14
        
        # --- Reward/Cost Parameters (CORRECTED to match official defaults) ---
        self.healthy_reward = 1.5
        self.forward_reward_weight = 1 
        self.ctrl_cost_weight = 0.5   
        self.contact_cost_weight = 5e-4
        
        self.contact_force_range = (-1.0, 1.0) 
        
        # --- Health Parameters ---
        self.healthy_z_range = (0.265, 1.2)

        self.reset_noise_scale = 0.1
        
        # --- Simulation Parameters ---
        self.frame_skip = 5
        # (CORRECTED) Calculate dt *after* loading model
        self.dt = self.model.opt.timestep * self.frame_skip

    def _get_obs(self):
        """
        To get the current observation from the simulation.
        """
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        
        position = position[2:]  # Exclude x, y
        
        # Get raw forces, skipping worldbody (geom 0)
        raw_contact_forces = self.data.cfrc_ext[1:]
        
        min_val, max_val = self.contact_force_range
        contact_force = np.clip(raw_contact_forces, min_val, max_val).flatten()
        
        return np.concatenate((position, velocity, contact_force)).astype(np.float64)

    def _is_healthy(self):
        """
        Check if the Ant is "healthy"
        """
        # 1. Check for finite state values
        # (Using self.data.qpos and qvel directly)
        is_finite = np.isfinite(self.data.qpos).all() and \
                    np.isfinite(self.data.qvel).all()
        
        # 2. Check z-coordinate (height)
        z_height = self.data.qpos[2]
        in_healthy_z_range = (
            self.healthy_z_range[0] <= z_height <= self.healthy_z_range[1]
        )
        
        return is_finite and in_healthy_z_range

    def _get_reward(self, x_velocity, action, is_healthy):
        """
        Calculates the reward components.
        """
        
        # 1. Healthy Reward
        healthy_reward = self.healthy_reward if is_healthy else 0.0
        
        # 2. Forward Reward
        forward_reward = self.forward_reward_weight * x_velocity
        
        # 3. Control Cost
        ctrl_cost = self.ctrl_cost_weight * np.square(action).sum()
        
        # 4. Contact Cost
        # Get forces for all geoms *except* worldbody (geom 0)
        robot_contact_forces = self.data.cfrc_ext[1:]
        clipped_contact_forces = np.clip(
            robot_contact_forces,
            self.contact_force_range[0],
            self.contact_force_range[1]
        )
        contact_cost = self.contact_cost_weight * np.square(clipped_contact_forces).sum()
        
        # Total Reward
        total_reward = healthy_reward + forward_reward - ctrl_cost - contact_cost
        
        # Dictionary for logging
        reward_info = {
            'reward_total': total_reward,
            'reward_healthy': healthy_reward,
            'reward_forward': forward_reward,
            'cost_ctrl': -ctrl_cost,
            'cost_contact': -contact_cost
        }
        
        return total_reward, reward_info

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        """
        
        # x-position of the main body
        x_position_before = self.data.qpos[0]
        
        # Apply action and step the simulation
        self.data.ctrl[:] = action

        #step the simulation
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        
        # --- Get values *after* the step ---
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt
        
        # --- Check Health (used for reward and termination) ---
        is_healthy = self._is_healthy()
        
        # --- Calculate Reward ---
        # Pass the new average x_velocity to the reward function
        reward, reward_info = self._get_reward(x_velocity, action, is_healthy)
        
        # --- Check Termination ---
        terminated = not is_healthy
        
        # --- Get Observation ---
        observation = self._get_obs()
        
        # --- Info Dictionary ---
        info = {
            'x_position': x_position_after,
            'distance_from_origin': np.linalg.norm(self.data.qpos[0:2], ord=2),
            'x_velocity': x_velocity,
            **reward_info  # Unpacks all reward components
        }
        
        # Truncated is False (no time limit, SB3/Gymnasium wrappers will add this)
        truncated = False
        
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        """
        # Set seed
        super().reset(seed=seed)
        
        # Reset to initial qpos and qvel with noise
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        # qpos = init_qpos + U([-noise, noise])
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        # qvel = init_qvel + N(0, noise^2)
        qvel = (
            self.init_qvel
            + self.reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        
        # Forward simulation to settle
        mujoco.mj_forward(self.model, self.data)
        
        observation = self._get_obs()
        
        info = {
            'x_position': self.data.qpos[0],
            'y_position': self.data.qpos[1],
            'distance_from_origin': np.linalg.norm(self.data.qpos[0:2], ord=2),
            # Note: initial velocities will be near-zero from noise
            'x_velocity': self.data.qvel[0], 
            'y_velocity': self.data.qvel[1]
        }
        
        if self.render_mode == "human":
            self.render()
            
        return observation, info

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode != "human":
            return
            
        if self.viewer is None:
            # Launch the passive viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Sync the viewer with the simulation data
        self.viewer.sync()

    def close(self):
        """
        Clean up resources (close the viewer).
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
