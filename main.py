import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import time
import numpy as np
from PIL import Image
import inspect
import xml.etree.ElementTree as ET
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from scipy.spatial.transform import Rotation


# --- 1. Custom Terrain and Environment Setup ---

def create_uneven_terrain(output_dir=".", size=(128, 128), max_height=0.5, terrain_size=(30, 30)):
    """
    Generates a heightfield PNG image and a custom MuJoCo XML file for the Ant environment.
    This version correctly modifies the existing asset block to avoid duplicate names.
    """
    image_path_rel = "terrain.png"
    image_path_abs = os.path.abspath(os.path.join(output_dir, image_path_rel))
    xml_path = os.path.join(output_dir, "ant_uneven.xml")

    # --- Step 1: Generate the heightfield image ---
    print(f"Generating heightfield image at {image_path_abs}...")
    x = np.linspace(-6 * np.pi, 6 * np.pi, size[1])
    y = np.linspace(-6 * np.pi, 6 * np.pi, size[0])
    xx, yy = np.meshgrid(x, y)
    base_wave = np.sin(xx) + np.sin(yy)
    sharp_wave = 0.5 * (np.sin(2.5 * xx) + np.sin(2.5 * yy))
    z = base_wave + sharp_wave
    height_data = (255 * (z - z.min()) / (z.max() - z.min())).astype(np.uint8)
    img = Image.fromarray(height_data, 'L')
    img.save(image_path_abs)
    print("Heightfield image generated.")

    # --- Step 2: Create the custom XML file by modifying the original ---
    print(f"Creating custom XML at {xml_path}...")
    gym_path = os.path.dirname(inspect.getfile(gym))
    original_xml_path = os.path.join(gym_path, "envs/mujoco/assets/ant.xml")

    tree = ET.parse(original_xml_path)
    root = tree.getroot()

    # Find worldbody and remove the original floor
    worldbody = root.find("worldbody")
    floor = worldbody.find("./geom[@name='floor']")
    if floor is not None:
        worldbody.remove(floor)
    else:
        raise ValueError("Could not find floor geom in ant.xml")

    # Add the new hfield geom to the worldbody.
    # It will use the 'MatPlane' material which should already be defined in the assets.
    ET.SubElement(
        worldbody, "geom",
        attrib={
            "conaffinity": "1", "condim": "3", "name": "terrain",
            "material": "MatPlane", "type": "hfield", "hfield": "terrain"
        },
    )

    # Find the existing asset tag. This is the key change.
    asset = root.find("asset")
    if asset is None:
        # If for some reason it doesn't exist, create it.
        asset = ET.Element("asset")
        root.insert(2, asset)

    # Add ONLY the hfield definition to the asset block.
    # The texture and material it uses ('texgeom', 'MatPlane') already exist.
    ET.SubElement(
        asset, "hfield",
        name="terrain", file=image_path_abs,
        size=f"{terrain_size[0]} {terrain_size[1]} {max_height} 0.1"
    )

    # Write the modified XML to a new file
    tree.write(xml_path, encoding="unicode")
    print("Custom XML created.")

    return os.path.abspath(xml_path)


# This wrapper modifies the reward to penalize sideways movement.
class CustomAntWrapper(gym.Wrapper):
    def __init__(self, env, penalty_weight=1.5, initial_pos=None):
        super().__init__(env)
        self.penalty_weight = penalty_weight
        
        # Only set up position modification if an initial_pos is provided
        self.initial_pos = None
        if initial_pos is not None:
            self.initial_pos = np.array(initial_pos)
            assert hasattr(self.env.unwrapped, "set_state"), "Environment must have a 'set_state' method to set position."
            assert hasattr(self.env.unwrapped, "_get_obs"), "Environment must have a '_get_obs' method to get new observation."

    def step(self, action):
        """
        Applies the action and modifies the reward to penalize sideways velocity.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Reward modification logic from StraightAntEnv
        y_velocity = info.get('y_velocity', 0)
        sideways_penalty = self.penalty_weight * np.abs(y_velocity)
        modified_reward = reward - sideways_penalty
        
        return obs, modified_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Resets the environment and then sets a custom initial position if one was specified.
        """
        # First, reset the environment to its default state
        obs, info = self.env.reset(**kwargs)

        # Position modification logic from ResetPositionWrapper
        # This part only runs if an initial_pos was provided during initialization
        if self.initial_pos is not None:
            qpos = self.env.unwrapped.data.qpos
            qvel = self.env.unwrapped.data.qvel
            
            # Overwrite the x, y, z position
            qpos[:3] = self.initial_pos
            
            # Set the modified state back into the simulation
            self.env.unwrapped.set_state(qpos, qvel)
            
            # The observation has changed, so get the new one
            obs = self.env.unwrapped._get_obs()

        return obs, info


class CustomAntEnv(AntEnv):
    """
    A single, configurable Ant environment that allows for:
    1. A custom healthy Z-range for termination.
    2. A custom orientation threshold for termination.
    """
    def __init__(self, healthy_z_range=(0.2, 1.0), orientation_threshold=0.0, **kwargs):
        """
        Initializes the environment with custom termination rules.

        Args:
            healthy_z_range (tuple): The (min, max) height for the ant's torso.
            orientation_threshold (float): The minimum value for the z-component of the
                                           torso's 'up' vector. The default is 0.0,
                                           a negative value like -0.5 is more lenient.
            **kwargs: Other arguments for the parent AntEnv.
        """
        # First, call the parent class's constructor
        super().__init__(**kwargs)
        
        # Store our custom termination parameters
        self._healthy_z_range = healthy_z_range
        self._orientation_threshold = orientation_threshold
        
        print(f"Custom Ant environment initialized with:")
        print(f"  - Healthy Z-Range: {self._healthy_z_range}")
        print(f"  - Orientation Threshold: {self._orientation_threshold}")

    @property
    def is_healthy(self):
        # 1. Height Check (unchanged)
        min_z, max_z = self._healthy_z_range
        is_within_height_range = self.data.qpos[2] >= min_z and self.data.qpos[2] <= max_z

        # 2. Orientation Check (updated to use scipy)
        q = self.data.qpos[3:7]  # Get quaternion in [w, x, y, z] format
        
        # Reorder quaternion to [x, y, z, w] for scipy
        q_scipy = q[[1, 2, 3, 0]] 
        
        # Create a rotation object and apply it to the 'up' vector [0, 0, 1]
        rot = Rotation.from_quat(q_scipy)
        rotated_up_vector = rot.apply([0, 0, 1])
        z_up = rotated_up_vector[2] # Extract the Z-component
        
        is_oriented = z_up > self._orientation_threshold

        return is_within_height_range and is_oriented

# --- 2. Setup ---
custom_xml_path = create_uneven_terrain()

log_dir = "/tmp/gym/"
model_dir = "models/PPO_Straight_Uneven"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "ppo_straight_ant_uneven")
gym.register(
    id='CustomAnt-v1',
    entry_point=__name__ + ':CustomAntEnv', # Points to our new combined class
    max_episode_steps=1000,
)


TRAIN_MODEL = False
start_pos = (0, 0, 0.5)

if TRAIN_MODEL:
    # --- 3. Training ---
    print("\nCreating custom uneven environment and starting training...")

    def make_env():
        env = gym.make('Ant-v4', xml_file=custom_xml_path)
        env = CustomAntWrapper(env, penalty_weight=1.5, initial_pos= start_pos)
        return env

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

    total_timesteps = 500000
    print(f"Training model for {total_timesteps} timesteps on uneven terrain...")
    model.learn(total_timesteps=total_timesteps)

    print(f"Training complete. Saving model to {model_path}")
    model.save(model_path)
    print("Model saved.")

    env.close()

# --- 4. Evaluation and Visualization ---
print("\n--- Loading and Evaluating Model on Uneven Terrain ---")

if not os.path.exists(model_path + ".zip"):
    print(f"Error: Model not found at {model_path}.zip")
    print("Please run the script with TRAIN_MODEL = True first.")
else:
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    print("Model loaded.")

    print("Creating evaluation environment.")
    # base_env = gym.make('Ant-v4', xml_file=custom_xml_path, render_mode='human')
    custom_env = gym.make('CustomAnt-v1', xml_file=custom_xml_path, render_mode='human',healthy_z_range=(-3.0, 3.0),orientation_threshold=-1)


    eval_env = CustomAntWrapper(
        custom_env, 
        penalty_weight=1.5, 
        initial_pos=start_pos
    )

    obs, info = eval_env.reset()

    print("Starting evaluation. The simulation window will open.")
    try:
        for _ in range(10000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            time.sleep(1/10)

            if terminated or truncated:
                print("Episode finished. Resetting environment.")
                obs, info = eval_env.reset()
                # break


    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        eval_env.close()
        print("Evaluation finished and environment closed.")