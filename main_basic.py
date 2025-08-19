import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import time

# --- 1. Setup ---
# Create directories to save logs and models
log_dir = "/tmp/gym/"
model_dir = "models/PPO"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Define the model filename
model_path = os.path.join(model_dir, "ppo_ant")

# Set to True to train a new model, False to load and view an existing one
TRAIN_MODEL = False

if TRAIN_MODEL:
    # --- 2. Training ---
    # Create the vectorized Ant environment.
    # Using a vectorized environment (make_vec_env) is recommended by Stable Baselines3
    # as it allows for parallel training and is more efficient.
    print("Creating environment and starting training...")
    env = make_vec_env('Ant-v5', n_envs=4) # Using 4 parallel environments

    # Instantiate the PPO agent with a Multi-Layer Perceptron (Mlp) policy.
    # The hyperparameters used here are a good starting point for many MuJoCo tasks.
    # - n_steps: The number of steps to run for each environment per update.
    # - batch_size: The mini-batch size for updating the policy.
    # - n_epochs: The number of epochs to run when updating the policy.
    # - gamma: The discount factor.
    # - gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
    # - clip_range: The clipping parameter for the PPO objective.
    # - ent_coef: Entropy coefficient for the loss calculation.
    # - verbose=1: Print out training information.
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

    # Train the agent. This can take a significant amount of time.
    # For a real training session, you would use a much larger number of timesteps,
    # e.g., 1,000,000 or more.
    # We use 200,000 here for a quicker demonstration.
    total_timesteps = 500000
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

    # Create a new, single environment for visualization.
    # The 'render_mode' is set to 'human' to see the simulation.
    eval_env = gym.make('Ant-v5', render_mode='human',width=1280, height=720)

    # Reset the environment to get the initial observation
    obs, info = eval_env.reset()

    print("Starting evaluation. The simulation window will open.")
    # Run the simulation loop
    # The agent will now use the learned policy to choose actions.
    try:
        for _ in range(5000): # Run for 5000 steps
            # Get the action from the model (deterministic=True means we take the best action)
            action, _states = model.predict(obs, deterministic=True)

            # Take the action in the environment
            obs, reward, terminated, truncated, info = eval_env.step(action)

            time.sleep (1/30)

            # If the episode is over (terminated or truncated), reset the environment
            if terminated or truncated:
                print("Episode finished. Resetting environment.")
                obs, info = eval_env.reset()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        # Close the environment window
        eval_env.close()
        print("Evaluation finished and environment closed.")

