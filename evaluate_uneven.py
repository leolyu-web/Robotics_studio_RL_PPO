import gymnasium
from stable_baselines3 import PPO
import time
import myant_env 
import numpy as np
import mujoco
from uneven_train import uneven_Wrapper, wrapper_kwargs

gymnasium.register(
     id='MyAnt-v0',
     entry_point='myant_env:myAntEnv',
)

if __name__ == "__main__":
    # --- Load the Trained Model ---
    model = PPO.load("PPO_model/ppo_myant_uneven_model.zip") # Make sure this is the correct model

    # --- Create and Wrap Evaluation Environment ---
    # create the base environment and use the XML file the model was trained on
    eval_env_base = gymnasium.make(
        'MyAnt-v0', 
        render_mode="human", 
        xml_path="assets/ant_uneven.xml"
    )
    
    # Now, manually wrap the environment
    eval_env = uneven_Wrapper(eval_env_base, **wrapper_kwargs)
    
    print("Starting evaluation with wrapper... Close the MuJoCo window to stop.")

    num_episodes = 1000
    for ep in range(num_episodes):
        # The wrapper's reset() method will be called
        obs, info = eval_env.reset() 
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            
            # The wrapper's step() method will be called
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            if terminated or truncated:
                print(f"Episode {ep+1} finished.")

            time.sleep(1/30)
            
    eval_env.close()