import gymnasium
from stable_baselines3 import PPO
import time
import myant_env 

gymnasium.register(
     id='MyAnt-v0',
     entry_point='myant_env:myAntEnv',
)

if __name__ == "__main__":
    # --- Load the Trained Model ---
    model = PPO.load("PPO_model/ppo_myant_model.zip")

    # --- Create Evaluation Environment ---
    eval_env = gymnasium.make('MyAnt-v0', render_mode="human")

    print("Starting evaluation... Close the MuJoCo window to stop.")

    num_episodes = 1000
    for ep in range(num_episodes):
        obs, info = eval_env.reset()
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            if terminated or truncated:
                print(f"Episode {ep+1} finished.")

            time.sleep(1/30)
    eval_env.close()




