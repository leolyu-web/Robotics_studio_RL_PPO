import gymnasium as gym
import os
import shutil
from gymnasium_robotics.envs.ant import AntEnv

# Load the Ant environment to access its properties
env = gym.make('Ant-v5')

# Get the full path to the XML file
xml_file_path = env.unwrapped.model_path
print(f"Original Ant XML is at: {xml_file_path}")

# It's good practice to work in a dedicated folder
custom_env_dir = "custom_envs"
os.makedirs(custom_env_dir, exist_ok=True)

# Define the path for your new XML and copy the original over
new_xml_path = os.path.join(custom_env_dir, "ant_hilly.xml")
shutil.copy(xml_file_path, new_xml_path)
print(f"Copied XML to: {new_xml_path}")

env.close()