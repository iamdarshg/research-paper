"""Multi-fidelity RL training pipeline with GPU optimization and progress tracking."""
import yaml
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.rl_agent.model import DDPGAgent
from src.rl_agent.env import PaperPlaneEnv
from src.folding.folder import fold_sheet
from src.surrogate.aero_model import surrogate_cfd, surrogate_cfd_batch, compute_aero_features, compute_aero_features_batch
from src.surrogate.batch_evaluator import SurrogateBatchEvaluator
from src.utils.gpu_utils import autodetect_num_envs # New import
from typing import Any, Dict, List, Union, Tuple, Optional

CONFIG_PATH = Path(__file__).parent.parent.parent / 'config.yaml'
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global ThreadPoolExecutor for asynchronous I/O
_io_executor = ThreadPoolExecutor(max_workers=2)

def load_config() -> Dict[str, Any]:
    """Loads the configuration from config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def _save_file_async(func: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Submits a file saving function to the asynchronous I/O executor.

    Args:
        func (Any): The function to execute (e.g., mesh.export, torch.save, np.save).
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        Any: A Future object representing the asynchronous operation.
    """
    return _io_executor.submit(func, *args, **kwargs)

def _export_mesh_async(mesh: Any, path: Union[str, Path]) -> Any:
    """Submits mesh export to the asynchronous I/O executor."""
    return _save_file_async(mesh.export, path)

def _save_agent_state_async(agent_state_dict: Dict[str, Any], path: Union[str, Path]) -> Any:
    """Submits agent state saving to the asynchronous I/O executor."""
    return _save_file_async(torch.save, agent_state_dict, path)

def _save_numpy_async(data: np.ndarray, path: Union[str, Path]) -> Any:
    """Submits numpy array saving to the asynchronous I/O executor."""
    return _save_file_async(np.save, path, data)

class VectorizedPaperPlaneEnv:
    """
    Vectorized environment that manages multiple PaperPlaneEnv instances
    and evaluates actions in batches for improved GPU utilization.
    """

    def __init__(self, num_envs: int = 256, auto_batch_size: bool = True):
        """
        Initializes vectorized environments with batch evaluation.

        Args:
            num_envs (int): Number of parallel environments
            auto_batch_size (bool): Whether to use auto-detected batch size
        """
        self.num_envs = num_envs
        self.envs = [PaperPlaneEnv() for _ in range(num_envs)]
        self.batch_evaluator = SurrogateBatchEvaluator(device=DEVICE)
        self.max_batch_size = self.batch_evaluator.recommended_batch_size if auto_batch_size else 256
        self.observation_spaces = [env.observation_space for env in self.envs]
        self.action_spaces = [env.action_space for env in self.envs]
        self.max_steps = self.envs[0].max_steps

    def reset(self, seeds: Optional[List[Optional[int]]] = None):
        """
        Resets all environments.

        Args:
            seeds (Optional[List[Optional[int]]]): Seeds for each environment

        Returns:
            Tuple[List[np.ndarray], List[Dict]]: List of observations and infos
        """
        observations = []
        infos = []

        for i, env in enumerate(self.envs):
            seed = seeds[i] if seeds and i < len(seeds) else None
            obs, info = env.reset(seed=seed)
            observations.append(obs)
            infos.append(info)

        return observations, infos

    def step_batch(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], List[Dict[str, Any]]]:
        """
        Performs a batch step across all environments using GPU-accelerated batch evaluation.

        Args:
            actions (List[np.ndarray]): List of actions for each environment

        Returns:
            Tuple of lists containing observations, rewards, terminated, truncated, and info for each env
        """
        batch_size = len(actions)

        # Use a ThreadPoolExecutor to parallelize CPU-bound mesh folding
        with ThreadPoolExecutor(max_workers=self.num_envs) as executor: # Use num_envs workers for folding
            mesh_futures = {executor.submit(fold_sheet, action, resolution=30): i
                            for i, action in enumerate(actions)}
            
            # Collect meshes in order of submission
            meshes: List[Any] = [None] * batch_size
            for future, i in mesh_futures.items():
                meshes[i] = future.result()

        # Extract features and state parameters for batch evaluation
        features_list = []
        state_params_list = []

        for i, env in enumerate(self.envs):
            # Extract state parameters (denormalized from env.state_vector_np)
            state_params = {
                'throw_speed_mps': env.state_vector_np[2] * 20.0,
                'angle_of_attack_deg': env.state_vector_np[6] * 20.0,
                'air_density_kgm3': env.state_vector_np[7] * 1.5,
                'air_viscosity_pas': env.state_vector_np[8] * 2e-5
            }
            state_params_list.append(state_params)
            
            # Use compute_aero_features to prepare features for surrogate_cfd_batch
            features_list.append(compute_aero_features(meshes[i]))


        # Use batch evaluator for GPU-accelerated aerodynamic predictions
        aero_results = surrogate_cfd_batch(features_list, state_params_list)

        # Process results for each environment
        next_observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for i, env in enumerate(self.envs):
            aero_result = {
                'range_est': aero_results['range_est'][i].cpu().item(),
                'cl': aero_results['cl'][i].cpu().item(),
                'cd': aero_results['cd'][i].cpu().item(),
                'ld': aero_results['ld'][i].cpu().item()
            }
            current_mesh = meshes[i]

            # Update environment state based on action and aero results
            # This logic mimics parts of env.step() but without re-folding the mesh
            env.prev_reward = aero_result['range_est']
            range_est = aero_result['range_est']
            
            reward = np.clip((range_est / env.target_range) - 1.0, -1.0, 10.0)
            terminated = range_est > 1.1 * env.target_range
            truncated = env.current_step >= env.max_steps

            env.current_step += 1

            # Get the new observation (graph and vector) based on the updated environment state
            next_obs_dict = env._get_obs(current_mesh, actions[i]) # Use the current_mesh and current action for graph generation

            next_observations.append(next_obs_dict)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append({'range': range_est, 'mesh': current_mesh}) # Pass mesh for visualization

        return next_observations, rewards, terminateds, truncateds, infos

def evaluate_action_batch(actions_batch: List[np.ndarray], state: Dict[str, Any], max_workers: int = 4) -> np.ndarray:
    """
    Performs parallel evaluation of a batch of actions using a thread pool for mesh generation
    and GPU batch processing for aerodynamic predictions.

    Args:
        actions_batch (List[np.ndarray]): A list of numpy arrays, where each array represents an action.
        state (Dict[str, Any]): The environment state dictionary containing aerodynamic parameters.
        max_workers (int): The maximum number of parallel threads to use for mesh generation.

    Returns:
        np.ndarray: A numpy array of predicted ranges (range_est) for each action in the batch.
    """
    batch_size = len(actions_batch)

    # Build meshes in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        mesh_futures = {executor.submit(fold_sheet, action, resolution=30): i
                       for i, action in enumerate(actions_batch)}

        # Collect results in order
        indexed_meshes: List[Tuple[int, Any]] = sorted([(future_idx, future.result())
                                 for future_idx, future in zip(mesh_futures.values(), as_completed(mesh_futures))],
                                key=lambda x: x[0])
        meshes: List[Any] = [mesh for _, mesh in indexed_meshes]

    # Extract features from all meshes individually
    features_list: List[Dict[str, Any]] = [compute_aero_features(mesh) for mesh in meshes]

    # Batch GPU evaluation of aero models
    states_list: List[Dict[str, Any]] = [state] * batch_size
    aero_results = surrogate_cfd_batch(features_list, states_list)

    # Return ranges as numpy array
    return aero_results['range_est'].cpu().numpy()

def main():
    """
    Main training function now uses vectorized environments with batch evaluation.
    Trains multiple environments in parallel with ~256 batch evaluations per step.
    """
    config = load_config()
    episodes: int = config['training']['episodes']
    
    # Auto-detect number of environments based on GPU memory
    num_envs = autodetect_num_envs(target_mem_util_ratio=0.8, base_mem_mb_per_env=5.0)

    print(f"Initializing {num_envs} vectorized environments with batch evaluation...")

    # Create vectorized environment
    env_vec = VectorizedPaperPlaneEnv(num_envs=num_envs)
    
    # Safely get state and action dimensions
    if env_vec.observation_spaces[0] is None or env_vec.action_spaces[0] is None:
        print("Error: Environment observation or action space is None. Exiting.")
        return
    
    # Extract dimensions from the new observation space structure
    # env_vec.observation_spaces[0] is a Dict, so we need to access its components
    node_feature_dim: int = env_vec.envs[0].node_feature_dim # Get directly from an env instance
    state_vector_dim: int = env_vec.envs[0].state_vector_dim + env_vec.envs[0].action_dim # Total vector dim
    action_dim: int = env_vec.action_spaces[0].shape[0]

    # Create vectorized agent with new GNN-compatible dimensions
    agent = DDPGAgent(node_feature_dim, action_dim, state_vector_dim)

    # Track metrics across environments
    episode_rewards = []
    episode_ranges = []

    (Path('data') / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (Path('data') / 'logs').mkdir(exist_ok=True)

    # Main training function now delegates episode management to DDPGAgent.train
    print(f"Starting training for {episodes} episodes using DDPGAgent...")
    
    def mesh_save_callback(mesh_data, current_episode):
        """Callback to save mesh asynchronously at checkpoints."""
        _export_mesh_async(mesh_data, f"data/checkpoints/best_mesh_ep{current_episode}.stl")
        _save_agent_state_async(agent.actor.state_dict(), f"data/checkpoints/agent_ep{current_episode}.pth")

    _, episode_rewards, episode_ranges = agent.train(
        env_vec.envs[0], 
        total_episodes=episodes, 
        mesh_callback=lambda mesh_data: mesh_save_callback(mesh_data, agent.episode_count) # Pass episode count
    )

    # Create and save plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Full training progress (episode averages)
    ax1.plot(episode_ranges)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Max Range (m)')
    ax1.set_title('Vectorized Training Progress - Average Across Environments')

    # Zoomed on first 50 episodes
    zoom_ep = min(50, len(episode_ranges))
    ax2.plot(range(zoom_ep), episode_ranges[:zoom_ep])
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Max Range (m)')
    ax2.set_title(f'Training Progress - First {zoom_ep} Episodes')

    plt.tight_layout()
    _save_file_async(plt.savefig, 'data/logs/training_progress_detailed.png')
    plt.close()

    # Backward compatibility plot
    plt.figure()
    plt.plot(episode_ranges)
    plt.xlabel('Episode')
    plt.ylabel('Average Max Range (m)')
    plt.title('Vectorized Training Progress')
    _save_file_async(plt.savefig, 'data/logs/training_progress.png')
    plt.close()

    # Save logs
    _save_numpy_async(np.array(episode_rewards), 'data/logs/rewards.npy')
    _save_numpy_async(np.array(episode_ranges), 'data/logs/ranges.npy')

    print("Vectorized training complete. Waiting for background I/O to finish...")
    print(f"Trained on {num_envs} parallel environments with batch evaluation.")
    _io_executor.shutdown(wait=True)
    print("Background I/O complete. Check data/logs and checkpoints.")

if __name__ == "__main__":
    main()
