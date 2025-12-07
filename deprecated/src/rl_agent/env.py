"""Custom Gymnasium env for paper airplane fold optimization."""
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from src.folding.folder import fold_sheet
from src.folding.sheet import load_config # Centralized config loader
from src.surrogate.aero_model import surrogate_cfd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import trimesh
from torch_geometric.data import Data # Import Data for graph representation
from torch_geometric.utils import from_networkx, to_undirected # For graph conversion

class PaperPlaneEnv(gym.Env):
    """
    Custom Gymnasium environment for optimizing paper airplane folds using RL.
    The environment simulates folding a paper plane and evaluates its aerodynamic performance.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30} # Define metadata

    def __init__(self, render_mode: Optional[str] = None):
        """
        Initializes the PaperPlaneEnv.

        Args:
            render_mode (Optional[str]): Not used for this environment, but required by Gymnasium API.
        """
        super().__init__()
        self.config = load_config() # Use the centralized load_config
        self.n_folds = self.config['project']['n_folds']
        self.action_dim = self.n_folds * 5  # x1,y1,x2,y2,angle for each fold
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        
        # State Vector: [w,h_norm, v_inf, temp, target_range, throw_v, aoa, rho, mu] norm [0,1]
        self.state_vector_dim = 9
        # Node features: 3D coordinates (x,y,z)
        self.node_feature_dim = 3 

        # The observation space is now a dictionary containing the graph and the state vector
        # We need to define placeholder shapes for these
        # The graph structure is dynamic, so we provide an example Data object's structure
        # The vector part is a Box.
        # This will need to be properly defined for Gymnasium's check_env later.
        # For now, it will return a dict { 'graph': Data, 'vector': np.ndarray }
        self.observation_space = spaces.Dict({
            "graph": spaces.Box(low=-np.inf, high=np.inf, shape=(100, self.node_feature_dim), dtype=np.float32), # Placeholder for max 100 nodes
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_vector_dim + self.action_dim,), dtype=np.float32)
        })
        
        self.target_range: float = self.config['goals']['target_range_m']
        self.current_step: int = 0
        self.max_steps: int = 200
        self.state_vector_np: np.ndarray # The non-graph part of the state
        self.prev_reward: float = 0.0 # Initialize prev_reward
        self.resolution = 30 # Default resolution for mesh folding

    def _mesh_to_graph_data(self, mesh: trimesh.Trimesh) -> Data:
        """Converts a trimesh object to a PyTorch Geometric Data object."""
        # Node features: vertex coordinates (x, y, z)
        x = torch.tensor(mesh.vertices, dtype=torch.float32)
        
        # Edge index: connectivity from mesh faces (triangles)
        # Convert faces to edge list, then to a unique, undirected edge_index
        edge_index = []
        for face in mesh.faces:
            for i in range(3):
                edge_index.append([face[i], face[(i + 1) % 3]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index)
        
        # We could add edge features (e.g., edge length) or other node attributes (e.g., normals, curvature)
        # For simplicity, starting with just vertex coordinates as node features.
        return Data(x=x, edge_index=edge_index)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Resets the environment to an initial state.

        Args:
            seed (Optional[int]): Seed for reproducibility.
            options (Optional[Dict[str, Any]]): Additional options for reset.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Initial observation (graph+vector) and info dictionary.
        """
        super().reset(seed=seed, options=options)
        self.current_step = 0
        
        # Initialize state_vector_np around config parameters
        state_vector_np = np.array([
            self.config['project']['sheet_width_mm'] / 300.0,  # normalized
            self.config['project']['sheet_height_mm'] / 300.0, # normalized
            self.config['goals']['throw_speed_mps'] / 20.0,    # normalized
            self.config['environment']['temperature_k'] / 300.0, # normalized
            self.target_range / 50.0,                          # normalized
            self.config['goals']['throw_speed_mps'] / 20.0,    # normalized
            self.config['goals']['angle_of_attack_deg'] / 20.0, # normalized
            self.config['environment']['air_density_kgm3'] / 1.5, # normalized
            self.config['environment']['air_viscosity_pas'] / 2e-5 # normalized
        ], dtype=np.float32)
        self.state_vector_np = state_vector_np

        # Create an initial "flat" paper plane mesh (or a simple representation)
        # This mesh won't change unless an action is applied.
        # For reset, we create a default flat sheet to get initial graph data
        default_action = np.full(self.action_dim, 0.5, dtype=np.float32) # A neutral/default action for initial mesh
        initial_mesh = fold_sheet(default_action, resolution=self.resolution)
        initial_graph_data = self._mesh_to_graph_data(initial_mesh)

        # Observation now includes both the graph and the state vector
        obs = {
            'graph': initial_graph_data,
            'vector': np.concatenate([self.state_vector_np, default_action]).astype(np.float32) # Include initial action in vector part
        }
        return self._get_obs(initial_mesh, default_action), {'mesh': initial_mesh} # Also return the mesh for visualization purposes

    def _get_obs(self, mesh: trimesh.Trimesh, action: np.ndarray) -> Dict[str, Any]:
        """
        Helper method to create an observation dictionary from a mesh and the current action.
        """
        graph_data = self._mesh_to_graph_data(mesh)
        # The vector part of the observation should include the current state vector and the action that led to this mesh
        vector_obs = np.concatenate([self.state_vector_np, action]).astype(np.float32)
        return {
            'graph': graph_data,
            'vector': vector_obs
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Performs one step in the environment given an action.

        Args:
            action (np.ndarray): The folding action vector.

        Returns:
            Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
                - next_obs (Dict[str, Any]): The next observation (graph+vector).
                - reward (float): The reward received.
                - terminated (bool): Whether the episode terminated.
                - truncated (bool): Whether the episode truncated.
                - info (Dict[str, Any]): Additional information.
        """
        # Fold mesh
        mesh = fold_sheet(action, resolution=self.resolution)
        
        # Convert mesh to graph data
        graph_data = self._mesh_to_graph_data(mesh)

        # Auto switch to CFD if surrogate range prediction deviates or low range
        surrogate_loss_level: float = 0.1 * self.target_range  # If predicted range < this, use CFD

        use_cfd: bool = self.prev_reward < surrogate_loss_level

        if use_cfd:
            from src.cfd.runner import run_openfoam_cfd
            if self.state_vector_np is None or self.state_vector_np.shape[0] < self.state_vector_dim:
                self.reset() # Re-initialize if state_vector_np is invalid
                print("WARNING: self.state_vector_np was invalid in step(), reset performed.")
            
            # Extract relevant state parameters for CFD
            cfd_state_params = {
                'throw_speed_mps': self.state_vector_np[2] * 20.0,
                'angle_of_attack_deg': self.state_vector_np[6] * 20.0,
                'air_density_kgm3': self.state_vector_np[7] * 1.5,
                'air_viscosity_pas': self.state_vector_np[8] * 2e-5
            }
            aero = run_openfoam_cfd(action, cfd_state_params, fidelity='high')
            range_est: float = aero['range_est']
        else:
            if self.state_vector_np is None or self.state_vector_np.shape[0] < self.state_vector_dim:
                self.reset() # Re-initialize if state_vector_np is invalid
                print("WARNING: self.state_vector_np was invalid in step(), reset performed.")

            # Extract relevant state parameters for surrogate
            surrogate_state_params = {
                'throw_speed_mps': self.state_vector_np[2] * 20.0,
                'angle_of_attack_deg': self.state_vector_np[6] * 20.0,
                'air_density_kgm3': self.state_vector_np[7] * 1.5,
                'air_viscosity_pas': self.state_vector_np[8] * 2e-5
            }
            aero = surrogate_cfd(mesh, surrogate_state_params)
            range_est: float = aero['range_est']

        self.prev_reward = range_est

        reward: float = np.clip((range_est / self.target_range) - 1.0, -1.0, 10.0)
        terminated: bool = range_est > 1.1 * self.target_range
        truncated: bool = self.current_step >= self.max_steps

        self.current_step += 1

        next_state_vector_np: np.ndarray = self.state_vector_np + np.random.normal(0, 0.05, self.state_vector_np.shape).astype(np.float32)
        self.state_vector_np = next_state_vector_np # Update the environment's state vector

        # The next observation is based on the current mesh and action, with the updated state_vector_np
        next_obs = self._get_obs(mesh, action)

        return next_obs, reward, terminated, truncated, {'range': range_est, 'mesh': mesh}
