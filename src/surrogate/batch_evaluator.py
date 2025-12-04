"""
Batch parallel surrogate model evaluator with GPU acceleration.
Enables efficient evaluation of multiple folded configurations simultaneously.
"""
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import trimesh # Re-adding missing import
from ..folding.folder import fold_sheet
from .aero_model import compute_aero_features, surrogate_cfd_batch
import yaml
from typing import Dict, Any, Union, List, Optional

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIG_PATH = Path(__file__).parent.parent.parent / 'config.yaml'

def load_config() -> Dict[str, Any]:
    """Loads the configuration from config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def _autodetect_batch_size(device: torch.device) -> int:
    """
    Automatically detects an appropriate GPU batch size based on available VRAM.
    Aims for a balance between utilization and avoiding OOM errors.

    Args:
        device (torch.device): The device (CPU or CUDA) to query memory for.

    Returns:
        int: Recommended batch size.
    """
    if device.type == 'cuda':
        total_memory_bytes = torch.cuda.get_device_properties(device).total_memory
        total_memory_gb = total_memory_bytes / (1024**3)
        
        # Heuristic: Adjust based on observed GPU memory usage for mesh processing and aero model
        if total_memory_gb >= 12: # e.g., high-end consumer GPUs (RTX 3080/4080, etc.)
            return 128
        elif total_memory_gb >= 6: # e.g., mid-range consumer GPUs (RTX 3060/4060, etc.)
            return 64
        else: # For smaller VRAM GPUs or conservative default
            return 32
    return 32 # Default for CPU or if auto-detection is off

class SurrogateBatchEvaluator:
    """GPU-accelerated batch evaluator for surrogate model predictions."""
    
    def __init__(self, device: torch.device = DEVICE, max_workers: int = 4, auto_batch_size: bool = True):
        """
        Initialize batch evaluator.
        
        Args:
            device (torch.device): The torch device (cuda or cpu) to perform computations on.
            max_workers (int): Number of parallel threads for CPU-bound mesh generation.
            auto_batch_size (bool): If True, automatically detect GPU batch size based on VRAM.
        """
        self.device = device
        self.max_workers = max_workers
        self.config = load_config()
        self.auto_batch_size = auto_batch_size
        self.recommended_batch_size: int
        if self.device.type == 'cuda' and self.auto_batch_size:
            self.recommended_batch_size = _autodetect_batch_size(self.device)
        else:
            self.recommended_batch_size = 32 # Default for CPU or if auto-detection is off
        
    def evaluate_batch(self, actions: Union[np.ndarray, List[np.ndarray]], 
                       state: Dict[str, Any], show_progress: bool = True, 
                       batch_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Evaluate multiple actions in parallel with GPU batch processing.
        
        Args:
            actions (Union[np.ndarray, List[np.ndarray]]): Numpy array of shape (N, action_dim) 
                                                           or list of action arrays.
            state (Dict[str, Any]): Dictionary with aerodynamic parameters 
                                    (e.g., air_density_kgm3, throw_speed_mps).
            show_progress (bool): Whether to show progress bars for mesh generation and GPU evaluation.
            batch_size (Optional[int]): GPU batch size for simultaneous aerodynamic evaluation. 
                                        If None and auto_batch_size is True, an automatically 
                                        detected batch size is used.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary with keys 'range_est', 'cl', 'cd', 'ld', 'Re'
                                   as numpy arrays of shape (N,).
        """
        if batch_size is None:
            if self.device.type == 'cuda' and self.auto_batch_size:
                batch_size = self.recommended_batch_size
            else:
                batch_size = 32 # Default if not auto-detecting or on CPU
        
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if isinstance(actions, np.ndarray):
            actions_list: List[np.ndarray] = [actions[i] for i in range(actions.shape[0])]
        else:
            actions_list = actions
        
        n_actions = len(actions_list)
        
        # Step 1: Parallel mesh generation with progress
        iterator = tqdm(actions_list, desc='Generating meshes', disable=not show_progress)
        meshes: List[trimesh.Trimesh] = [] # Type hint for meshes
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            mesh_futures = {executor.submit(fold_sheet, action, resolution=30): i 
                           for i, action in enumerate(iterator)}
            meshes_dict: Dict[int, trimesh.Trimesh] = {} # Type hint
            
            for future in as_completed(mesh_futures):
                idx = mesh_futures[future]
                meshes_dict[idx] = future.result()
            
            meshes = [meshes_dict[i] for i in range(n_actions)]
        
        # Step 2: GPU batch evaluation with progress
        all_results: Dict[str, np.ndarray] = { # Type hint for all_results
            'range_est': np.zeros(n_actions, dtype=np.float32),
            'cl': np.zeros(n_actions, dtype=np.float32),
            'cd': np.zeros(n_actions, dtype=np.float32),
            'ld': np.zeros(n_actions, dtype=np.float32),
            'Re': np.zeros(n_actions, dtype=np.float32)
        }
        
        iterator = tqdm(range(0, n_actions, batch_size), 
                       desc=f'GPU batch evaluation (Batch Size: {batch_size})', 
                       disable=not show_progress)
        
        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, n_actions)
            batch_meshes = meshes[batch_start:batch_end]
            
            # Extract features from meshes
            features_list: List[Dict[str, Any]] = [compute_aero_features(mesh) for mesh in batch_meshes]
            states_list_batch: List[Dict[str, Any]] = [state] * len(features_list) # Renamed to avoid confusion
            
            # GPU batch evaluation
            aero_results = surrogate_cfd_batch(features_list, states_list_batch)
            
            # Store results
            all_results['range_est'][batch_start:batch_end] = aero_results['range_est'].cpu().numpy()
            all_results['cl'][batch_start:batch_end] = aero_results['cl'].cpu().numpy()
            all_results['cd'][batch_start:batch_end] = aero_results['cd'].cpu().numpy()
            all_results['ld'][batch_start:batch_end] = aero_results['ld'].cpu().numpy()
            all_results['Re'][batch_start:batch_end] = aero_results['Re'].cpu().numpy()
        
        return all_results
    
    def evaluate_single(self, action: np.ndarray, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a single action (slow path for comparison).
        
        Args:
            action (np.ndarray): Numpy array of action parameters.
            state (Dict[str, Any]): Dictionary with aerodynamic parameters.
        
        Returns:
            Dict[str, float]: Dictionary with aero results (range_est, cl, cd, ld, Re).
        """
        mesh = fold_sheet(action, resolution=30)
        features = compute_aero_features(mesh)
        states_list_single: List[Dict[str, Any]] = [state] # Renamed
        features_list_single: List[Dict[str, Any]] = [features] # Renamed
        
        aero_results = surrogate_cfd_batch(features_list_single, states_list_single)
        
        return {
            'range_est': float(aero_results['range_est'].cpu().numpy()[0]),
            'cl': float(aero_results['cl'].cpu().numpy()[0]),
            'cd': float(aero_results['cd'].cpu().numpy()[0]),
            'ld': float(aero_results['ld'].cpu().numpy()[0]),
            'Re': float(aero_results['Re'].cpu().numpy()[0])
        }

def evaluate_population(actions: np.ndarray, state: Dict[str, Any], 
                        num_workers: int = 4, batch_size: Optional[int] = None, 
                        show_progress: bool = True) -> Dict[str, np.ndarray]:
    """
    Convenience function to evaluate a population of actions.
    
    Args:
        actions (np.ndarray): Numpy array of shape (N, action_dim).
        state (Dict[str, Any]): Dictionary with state parameters.
        num_workers (int): Number of parallel workers for mesh generation.
        batch_size (Optional[int]): GPU batch size. If None, auto-detected batch size is used.
        show_progress (bool): Whether to show progress bars.
    
    Returns:
        Dict[str, np.ndarray]: Dictionary of results with numpy arrays (range_est, cl, cd, ld, Re).
    """
    evaluator = SurrogateBatchEvaluator(max_workers=num_workers, device=DEVICE) # Pass device
    return evaluator.evaluate_batch(actions, state, show_progress=show_progress, batch_size=batch_size)
