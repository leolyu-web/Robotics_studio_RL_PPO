import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import myant_env 

# --- 1. Register the Environment ---
gymnasium.register(
     id='MyAnt-v0',
     entry_point='myant_env:myAntEnv' # The format is 'filename:ClassName'

)
    
if __name__ == '__main__':
    env_id = 'MyAnt-v0'
    
    # --- 2. Create Vectorized Environment ---
    num_cpu = 4  
    env = make_vec_env(env_id, n_envs=num_cpu)
    
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
    
    # --- 4. Train the Model ---
    print("Starting model training...")
    model.learn(total_timesteps=1000000)
    
    # --- 5. Save the Model ---
    print("Training finished. Saving model...")
    model.save("PPO_model/ppo_myant_model")
    
    env.close()

