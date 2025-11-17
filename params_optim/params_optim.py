import mujoco
import numpy as np
import time
import os
import mujoco.viewer
import json 
import matplotlib.pyplot as plt 
import sys 
import argparse 

# base path
MODEL_PATH = "assets/ant_pos_ctrl.xml"
PARAMS_DIR = "params_optim/params"

# set the path for model training
id = 0
PARAMS_FILE_RS = os.path.join(PARAMS_DIR, f"random_best_params_{id}.json")
PARAMS_FILE_HC = os.path.join(PARAMS_DIR, f"hill_climb_best_params_{id}.json")
PARAMS_FILE_COMBO = os.path.join(PARAMS_DIR, f"combined_best_params_{id}.json")

# set the path for model visualize
params_file_map = {
        "random": "params_optim/params/random_best_params.json",
        "hill": "params_optim/params/hill_climb_best_params.json",
        "combined": "params_optim/params/combined_best_params.json"
        }

# Define path for the comparison plot
PLOT_FILE_COMPARISON = os.path.join(PARAMS_DIR, "comparison_learning_curve.png")

CTRL_MIN = np.array([
    -0.5236,  # hip_4 (BR)
    0.5236,   # ankle_4 (BR)
    -0.5236,  # hip_1 (FL)
    0.5236,   # ankle_1 (FL)
    -0.5236,  # hip_2 (FR)
    -1.2217,  # ankle_2 (FR)
    -0.5236,  # hip_3 (BL)
    -1.2217   # ankle_3 (BL)
])

CTRL_MAX = np.array([
    0.5236,   # hip_4 (BR)
    1.2217,   # ankle_4 (BR)
    0.5236,   # hip_1 (FL)
    1.2217,   # ankle_1 (FL)
    0.5236,   # hip_2 (FR)
    -0.5236,  # ankle_2 (FR)
    0.5236,   # hip_3 (BL)
    -0.5236   # ankle_3 (BL)
])

# Parameters for our CPG (Central Pattern Generator)
NUM_PARAMS = 1 + 8 + 8 + 8  # 25 parameters
print(f"Using a CPG controller with {NUM_PARAMS} parameters.")

# Simulation parameters
EPISODE_LENGTH_SECONDS = 30
EPISODE_LENGTH_STEPS = int(EPISODE_LENGTH_SECONDS / 0.01)  # 0.01 is timestep from MJCF

# Hyperparameter for Hill Climber
HILL_CLIMBER_NOISE_STD = 0.3  # Standard deviation of the perturbation noise
# --- Parameters for adaptive noise ---
HILL_CLIMBER_ADAPTATION_THRESHOLD = 500  # Steps without improvement to trigger adaptation
HILL_CLIMBER_MIN_NOISE_STD = 0.001      # Minimum noise level

# Total iterations for comparison
TOTAL_ITERATIONS = 20000

def get_action_from_params(params, t):
    """
    Calculates actuator commands from a 25-parameter CPG.
    
    - params[0]:     global_frequency
    - params[1:9]:   8 offsets
    - params[9:17]:  8 amplitudes
    - params[17:25]: 8 phase_shifts
    """
    
    # Unpack parameters
    frequency = np.abs(params[0])
    offsets = params[1:9]
    amplitudes = np.abs(params[9:17])
    phases = params[17:25]
    
    # Calculate the common phase for all actuators
    base_phase = 2 * np.pi * frequency * t
    
    # This is fully vectorized:
    action = offsets + amplitudes * np.sin(base_phase + phases)
    
    # Clip the actions to be within the valid control range
    action_clipped = np.clip(action, CTRL_MIN, CTRL_MAX)
    return action_clipped

def evaluate_policy(model, data, params):
    """
    Runs one simulation episode and returns a reward:
    """
    # Reset the simulation to the initial state
    mujoco.mj_resetData(model, data)
    # Set the initial joint positions from the 'init_qpos' custom numeric data
    init_qpos = model.numeric('init_qpos').data
    data.qpos[0:len(init_qpos)] = init_qpos
    
    torso_id = model.body('torso').id
    
    # We must call mj_forward() after setting qpos to update 
    # derived quantities (like body positions) before the first step.
    mujoco.mj_forward(model, data)
    
    # --- Get initial X position for final bonus ---
    initial_x_position = data.body(torso_id).xpos[0]

    for _ in range(EPISODE_LENGTH_STEPS):
        # Get action from our controller
        action = get_action_from_params(params, data.time)
        data.ctrl[:] = action
        
        # Step the simulation
        mujoco.mj_step(model, data)
        
    # --- Calculate Final Reward ---
    
    # 1. Get the final x position
    final_x_position = data.body(torso_id).xpos[0]
    
    # 2. Calculate the bonus for total distance traveled
    distance_traveled = final_x_position - initial_x_position
    
    reward = distance_traveled

    return reward


def save_params(params, save_path):
    """Saves the given parameters to a JSON file."""
    
    # Get the directory part of the save_path
    save_dir = os.path.dirname(save_path)
    # Create the directory if it doesn't exist (e.g., "params/")
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    print(f"Saving parameters to {save_path}...")
    # Convert numpy array to a standard python list for JSON serialization
    params_as_list = params.tolist()
    with open(save_path, 'w') as f:
        json.dump(params_as_list, f, indent=4)
    print("Parameters saved.")

def visualize(params):
    """Visualizes the best-found policy."""
    
    if mujoco.viewer is None:
        print("Cannot visualize without mujoco-viewer.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model for visualization: {e}")
        return

    # Reset to initial state
    mujoco.mj_resetData(model, data)
    init_qpos = model.numeric('init_qpos').data
    data.qpos[0:len(init_qpos)] = init_qpos

    print("\nLaunching viewer to show best policy...")
    print("Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            sim_time = data.time
            
            # Get action from our controller
            action = get_action_from_params(params, sim_time)
            data.ctrl[:] = action
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Sync the viewer to the simulation time
            viewer.sync()
            
            # Wait if necessary to maintain real-time simulation speed
            elapsed_wall_clock = time.time() - start_time
            if sim_time > elapsed_wall_clock:
                time.sleep(sim_time - elapsed_wall_clock)


# --- Training Algorithms ---
def run_random_search(num_iterations, model, data):
    """Runs the random search algorithm and returns results."""
    print(f"Optimizing {NUM_PARAMS} parameters for a {EPISODE_LENGTH_SECONDS}-second episode.")

    # --- Initial Best Guess (25 params) ---
    best_params = np.zeros(NUM_PARAMS)
    best_params[0] = 1.0
    best_params[1:9] = (CTRL_MIN + CTRL_MAX) / 2.0
    best_params[9:17] = 0.1
    best_params[17:25] = 0.0
    best_reward = evaluate_policy(model, data, best_params)
    print(f"Initial policy reward: {best_reward:.2f}")

    reward_history = [best_reward]

    for i in range(num_iterations):
        candidate_params = np.zeros(NUM_PARAMS)
        
        # --- Smart Parameter Sampling (for 25 params) ---
        candidate_params[0] = 1.0 # fixed frequency
        offsets = np.zeros(8)
        amplitudes = np.zeros(8)
        phases = np.zeros(8)

        for j in range(8):
            actuator_min = CTRL_MIN[j]
            actuator_max = CTRL_MAX[j]
            total_range = actuator_max - actuator_min
            max_amplitude = total_range / 2.0
            
            amp = np.random.uniform(0.1, max_amplitude)
            amplitudes[j] = amp
            phases[j] = np.random.uniform(0, 2 * np.pi)
            
            # Ensure offset_min is not greater than offset_max
            offset_min_val = actuator_min + amp
            offset_max_val = actuator_max - amp
                
            offsets[j] = np.random.uniform(offset_min_val, offset_max_val)

        candidate_params[1:9] = offsets
        candidate_params[9:17] = amplitudes
        candidate_params[17:25] = phases
         
        candidate_reward = evaluate_policy(model, data, candidate_params)
        
        if candidate_reward > best_reward:
            best_reward = candidate_reward
            best_params = candidate_params

            print(f"  Iteration {i+1}/{num_iterations} | New Best Reward: {best_reward:.2f}")
        
        reward_history.append(best_reward)
            
    print(f"Final best reward: {best_reward:.2f}")
    return best_params, reward_history

def run_hill_climber(num_iterations, model, data, initial_params=None):
    """Runs the Hill Climber algorithm and returns results."""
    
    # --- MODIFIED: Use a local variable for noise std ---
    current_noise_std = HILL_CLIMBER_NOISE_STD
    print(f"Using initial noise standard deviation: {current_noise_std:.4f}")
    print(f"Optimizing {NUM_PARAMS} parameters for a {EPISODE_LENGTH_SECONDS}-second episode.")

    # --- Initial Best Guess (25 params) ---
    if initial_params is None:
        best_params = np.zeros(NUM_PARAMS)
        best_params[0] = 1.0
        best_params[1:9] = (CTRL_MIN + CTRL_MAX) / 2.0
        best_params[9:17] = 0.1
        best_params[17:25] = 0.0
    else:
        print("Starting from provided initial parameters.")
        best_params = initial_params
    
    best_reward = evaluate_policy(model, data, best_params)
    print(f"Initial policy reward: {best_reward:.2f}")

    reward_history = [best_reward]

    # --- NEW: Counter for non-improvement ---
    iterations_without_improvement = 0

    for i in range(num_iterations):
        
        # 1. Generate random noise (a "step")
        noise = np.random.normal(
            loc=0.0,  # Mean of the noise
            scale=current_noise_std,  # --- MODIFIED: Use adaptable variable ---
            size=NUM_PARAMS
        )
        
        # 2. Create the new candidate by perturbing the best parameters
        candidate_params = best_params + noise
         
        candidate_reward = evaluate_policy(model, data, candidate_params)
        
        if candidate_reward > best_reward:
            best_reward = candidate_reward
            best_params = candidate_params
            print(f"  Iteration {i+1}/{num_iterations} | New Best Reward: {best_reward:.2f}")
            # --- Reset counter on improvement ---
            iterations_without_improvement = 0
        else:
            # --- Increment counter on no improvement ---
            iterations_without_improvement += 1
        
        # --- Check if we need to adapt the noise ---
        if iterations_without_improvement >= HILL_CLIMBER_ADAPTATION_THRESHOLD:
            # Calculate new noise, ensuring it doesn't go below the minimum
            new_noise_std = max(current_noise_std * 0.75, HILL_CLIMBER_MIN_NOISE_STD)
            
            if new_noise_std == current_noise_std:
                pass
            else:
                current_noise_std = new_noise_std
                print(f"  Iteration {i+1} | No improvement for {HILL_CLIMBER_ADAPTATION_THRESHOLD} steps. Decreasing noise std to {current_noise_std:.4f}")
            
            # Reset counter after adaptation to give new noise a chance
            iterations_without_improvement = 0 
            
        reward_history.append(best_reward)
            
    print(f"Final best reward: {best_reward:.2f}")
    return best_params, reward_history


# --- Main Execution Logic ---

def train_and_compare_all(num_iterations):
    """
    Runs all three training modes (RS, HC, Combo), saves all parameters,
    and saves a comparison plot.
    """
    
    print("=== Starting Full Comparison Run ===")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 1. Random Search Only ---
    print("\n--- Running: Random Search Only ---")
    start_time = time.time()
    rs_params, rs_history = run_random_search(num_iterations, model, data)
    save_params(rs_params, PARAMS_FILE_RS)
    print("random_search:" , time.time()-start_time )

    # --- 2. Hill Climber Only ---
    print("\n--- Running: Hill Climber Only ---")
    start_time = time.time()
    hc_params, hc_history = run_hill_climber(num_iterations, model, data)
    save_params(hc_params, PARAMS_FILE_HC)
    print("hill_climber:" , time.time()-start_time )

    # --- 3. Combined (RS + HC) ---
    print("\n--- Running: Combined (RS + HC) ---")
    rs_iters = num_iterations // 2
    hc_iters = num_iterations - rs_iters
    
    print(f"Running Random Search for {rs_iters} iterations...")
    start_time = time.time()
    combo_rs_params, combo_rs_history = run_random_search(rs_iters, model, data)
    
    print(f"Running Hill Climber for {hc_iters} iterations...")
    combo_hc_params, combo_hc_history = run_hill_climber(
        hc_iters, 
        model, 
        data, 
        initial_params=combo_rs_params
    )
    
    # Combine the histories. Skip the first element of hc_history
    # as it's the same as the last element of rs_history.
    combined_history = combo_rs_history + combo_hc_history[1:]
    save_params(combo_hc_params, PARAMS_FILE_COMBO)
    print("combined:" , time.time()-start_time )

    print(f"\nPlotting and saving comparison to {PLOT_FILE_COMPARISON}...")
    plt.figure(figsize=(12, 8))
    plt.plot(rs_history, label="Random Search Only")
    plt.plot(hc_history, label="Hill Climber Only")
    plt.plot(combined_history, label=f"Combined ({rs_iters} RS + {hc_iters} HC)", linewidth=2.5)
    
    plt.title(f"Training Comparison ({num_iterations} Iterations)")
    plt.xlabel("Iteration")
    plt.ylabel("Best Reward")
    plt.legend()
    plt.grid(True)
    
    # Ensure params directory exists for the plot
    if not os.path.exists(PARAMS_DIR):
        os.makedirs(PARAMS_DIR)
        
    plt.savefig(PLOT_FILE_COMPARISON)
    print("Plot saved.")
    
    # Clean up model and data
    del model
    del data
    
    print("\n=== Comparison Run Complete ===")

def train_single_mode(mode, num_iterations):
    """Runs a single training mode and saves its parameters."""
    
    print(f"=== Starting Single Train Run: {mode.upper()} ===")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    if mode == "random":
        params, history = run_random_search(num_iterations, model, data)
        save_params(params, PARAMS_FILE_RS)
    elif mode == "hill":
        params, history = run_hill_climber(num_iterations, model, data)
        save_params(params, PARAMS_FILE_HC)
    elif mode == "combined":
        rs_iters = num_iterations // 2
        hc_iters = num_iterations - rs_iters
        print(f"Running Random Search for {rs_iters} iterations...")
        combo_rs_params, _ = run_random_search(rs_iters, model, data)
        print(f"Running Hill Climber for {hc_iters} iterations...")
        params, history = run_hill_climber(
            hc_iters, 
            model, 
            data, 
            initial_params=combo_rs_params
        )
        save_params(params, PARAMS_FILE_COMBO)
    else:
        print(f"Unknown train mode: {mode}")

    del model
    del data
    print(f"\n=== {mode.upper()} Training Complete ===")


def load_and_visualize(mode, params_file_map):
    """
    This is the evaluation function
    It loads specified parameters from a file and runs the visualization.
    """
    
    params_file = params_file_map.get(mode)
    
    if params_file is None:
        print(f"Error: Unknown visualize mode '{mode}'.")
        print("Choose from: random, hill, combined")
        return

    if not os.path.exists(params_file):
        print(f"Error: Parameters file not found at {params_file}")
        print(f"Please run 'python {sys.argv[0]} train' or 'python {sys.argv[0]} train {mode}' first.")
        sys.exit(1)
    
    print(f"Visualize mode: Loading parameters from {params_file}...")
    try:
        with open(params_file, 'r') as f:
            params_list = json.load(f)
        
        best_policy_params = np.array(params_list)
        
        if best_policy_params.shape[0] != NUM_PARAMS:
             print(f"Error: Loaded params have wrong shape. Expected {NUM_PARAMS}, got {best_policy_params.shape[0]}")
             sys.exit(1)
        
        # Call the main visualization function
        visualize(best_policy_params)
            
    except Exception as e:
        print(f"Error loading or parsing {params_file}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # --- argparse Logic ---
    parser = argparse.ArgumentParser(
        description="Train and visualize a CPG controller for an Ant robot.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(
        dest="command", 
        required=True,
        help="Main command to execute"
    )

    parser_train = subparsers.add_parser(
        "train", 
        help="Run training",
        description=(
            "Runs the training process.\n"
            "By default, runs all 3 modes (random, hill, combined) and saves a comparison plot.\n"
            "If a mode is specified, runs only that mode."
        )
    )
    parser_train.add_argument(
        "mode", 
        nargs="?",  
        default=None,
        choices=["random", "hill", "combined"],
        help="Optional: specific mode to train (default: run all)"
    )

    # 4. Create the 'visualize' subparser
    parser_visualize = subparsers.add_parser(
        "visualize", 
        help="Visualize a trained policy",
        description="Loads and visualizes a policy from a saved parameter file."
    )
    parser_visualize.add_argument(
        "mode", 
        nargs="?", 
        default="combined", 
        choices=["random", "hill", "combined"],
        help="Mode to visualize (default: combined)"
    )
    args = parser.parse_args()


    if args.command == "train":
        if args.mode is None:
            # User just typed "python script.py train"
            train_and_compare_all(TOTAL_ITERATIONS)
            print(f"\nTo visualize the best (combined) results, run:\n  python {sys.argv[0]} visualize combined")
        else:
            # User typed "python script.py train [mode]"
            train_single_mode(args.mode, TOTAL_ITERATIONS)
            print(f"\nTo visualize the results, run:\n  python {sys.argv[0]} visualize {args.mode}")

    elif args.command == "visualize":
        # User typed "python script.py visualize [mode]" or "visualize"
        # 'args.mode' will be 'combined' by default if not specified
        load_and_visualize(args.mode, params_file_map )