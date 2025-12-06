"""Comprehensive test suite for paper airplane optimization framework."""
import sys
from pathlib import Path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

import pytest
import numpy as np
import torch
import trimesh
from typing import Dict

# Import modules to test
from src.folding.folder import fold_sheet
from src.folding.sheet import create_sheet, load_config
from src.surrogate.aero_model import surrogate_cfd, compute_aero_features
from src.surrogate.gnn_surrogate import (
    GNNAeroSurrogate, mesh_to_graph, gnn_surrogate_cfd
)
from src.cfd.fluidx3d_runner import run_fluidx3d_cfd, fluidx3d_available
from src.rl_agent.env import PaperPlaneEnv
from src.rl_agent.model import DDPGAgent


class TestFolding:
    """Test folding simulation."""
    
    def test_create_sheet(self):
        """Test sheet creation."""
        mesh = create_sheet(resolution=20)
        assert mesh is not None
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        # Verify flat (z ~= 0)
        assert np.max(np.abs(mesh.vertices[:, 2])) < 1e-6
    
    def test_fold_sheet(self):
        """Test folding operation."""
        mesh = create_sheet(resolution=20)
        action = np.array([0.25, 0.25, 0.75, 0.25, 0.5, 0.3, 0.3, 0.7, 0.7, 0.3, 0.5])
        folded = fold_sheet(action, resolution=20)
        
        # Verify folded mesh is 3D
        z_range = np.max(mesh.vertices[:, 2]) - np.min(mesh.vertices[:, 2])
        z_range_folded = np.max(folded.vertices[:, 2]) - np.min(folded.vertices[:, 2])
        assert z_range_folded > z_range, "Folded mesh should be 3D"
    
    def test_fold_determinism(self):
        """Test that folding is deterministic."""
        action = np.random.uniform(0, 1, 25)
        mesh1 = fold_sheet(action, resolution=20)
        mesh2 = fold_sheet(action, resolution=20)
        
        assert np.allclose(mesh1.vertices, mesh2.vertices), "Folding should be deterministic"


class TestSurrogate:
    """Test aerodynamic surrogate models."""
    
    @pytest.fixture
    def mesh(self):
        """Fixture: folded mesh."""
        action = np.array([0.25, 0.25, 0.75, 0.25, 0.5, 0.3, 0.3, 0.7, 0.7, 0.3, 0.5])
        return fold_sheet(action, resolution=30)
    
    @pytest.fixture
    def state(self):
        """Fixture: environmental state."""
        config = load_config()
        return {
            'angle_of_attack_deg': config['goals']['angle_of_attack_deg'],
            'throw_speed_mps': config['goals']['throw_speed_mps'],
            'air_density_kgm3': config['environment']['air_density_kgm3'],
            'air_viscosity_pas': config['environment']['air_viscosity_pas']
        }
    
    def test_classical_surrogate(self, mesh, state):
        """Test classical surrogate."""
        result = surrogate_cfd(mesh, state)
        
        assert 'cl' in result
        assert 'cd' in result
        assert 'range_est' in result
        
        # Verify reasonable values
        assert -2 <= result['cl'] <= 2
        assert 0.01 <= result['cd'] <= 1
        assert 0 <= result['range_est'] <= 100
    
    def test_aero_features(self, mesh):
        """Test aerodynamic feature extraction."""
        features = compute_aero_features(mesh)
        
        assert 'area_proj' in features
        assert 'span' in features
        assert 'AR' in features
        assert 'camber' in features
        
        # Verify torch tensors on appropriate device
        assert isinstance(features['AR'], torch.Tensor)


class TestGNNSurrogate:
    """Test GNN-based surrogate."""
    
    @pytest.fixture
    def gnn_model(self):
        """Fixture: GNN model."""
        return GNNAeroSurrogate(
            node_feature_dim=7,
            edge_feature_dim=4,
            global_feature_dim=4,
            hidden_dim=64,
            num_layers=3,
            output_dim=3
        )
    
    @pytest.fixture
    def mesh(self):
        """Fixture: folded mesh."""
        action = np.array([0.25, 0.25, 0.75, 0.25, 0.5, 0.3, 0.3, 0.7, 0.7, 0.3, 0.5])
        return fold_sheet(action, resolution=30)
    
    @pytest.fixture
    def state(self):
        """Fixture: environmental state."""
        config = load_config()
        return {
            'angle_of_attack_deg': config['goals']['angle_of_attack_deg'],
            'throw_speed_mps': config['goals']['throw_speed_mps'],
            'air_density_kgm3': config['environment']['air_density_kgm3'],
            'air_viscosity_pas': config['environment']['air_viscosity_pas']
        }
    
    def test_mesh_to_graph(self, mesh, state):
        """Test mesh-to-graph conversion."""
        data = mesh_to_graph(mesh, state, device='cpu')
        
        assert data.x is not None  # Node features
        assert data.edge_index is not None  # Edges
        assert data.edge_attr is not None  # Edge features
        assert data.u is not None  # Global features
        
        # Verify shapes
        assert data.x.shape[1] == 7  # 7 node features
        assert data.edge_attr.shape[1] == 4  # 4 edge features
        assert data.u.shape[1] == 4  # 4 global features
    
    def test_gnn_forward(self, gnn_model, mesh, state):
        """Test GNN forward pass."""
        data = mesh_to_graph(mesh, state, device='cpu')
        
        with torch.no_grad():
            output = gnn_model(data)
        
        assert output.shape == (1, 3)  # Batch size 1, output dim 3
        
        # Verify reasonable ranges
        cl, cd, range_est = output[0].numpy()
        assert -2 <= cl <= 2
        assert 0.01 <= cd <= 1
        assert 0 <= range_est <= 100
    
    def test_gnn_surrogate_cfd(self, gnn_model, mesh, state):
        """Test GNN surrogate prediction."""
        result = gnn_surrogate_cfd(mesh, state, gnn_model)
        
        assert 'cl' in result
        assert 'cd' in result
        assert 'range_est' in result
        
        # Verify clipping
        assert -2 <= result['cl'] <= 2
        assert 0.01 <= result['cd'] <= 1
        assert 0 <= result['range_est'] <= 100


class TestFluidX3D:
    """Test FluidX3D integration."""
    
    def test_fluidx3d_availability(self):
        """Test FluidX3D availability check."""
        available = fluidx3d_available()
        assert isinstance(available, bool)
    
    @pytest.mark.skipif(not fluidx3d_available(), reason="FluidX3D not installed")
    def test_fluidx3d_cfd(self):
        """Test FluidX3D CFD runner (requires installation)."""
        action = np.array([0.25, 0.25, 0.75, 0.25, 0.5, 0.3, 0.3, 0.7, 0.7, 0.3, 0.5])
        mesh = fold_sheet(action, resolution=30)
        
        config = load_config()
        state = {
            'angle_of_attack_deg': config['goals']['angle_of_attack_deg'],
            'throw_speed_mps': config['goals']['throw_speed_mps'],
            'air_density_kgm3': config['environment']['air_density_kgm3'],
            'air_viscosity_pas': config['environment']['air_viscosity_pas']
        }
        
        result = run_fluidx3d_cfd(mesh, state)
        
        assert 'cl' in result
        assert 'cd' in result
        assert 'range_est' in result


class TestRLEnvironment:
    """Test RL environment."""
    
    def test_env_creation(self):
        """Test environment initialization."""
        env = PaperPlaneEnv()
        
        assert env.observation_space is not None
        assert env.action_space is not None
        
    def test_env_reset(self):
        """Test environment reset."""
        env = PaperPlaneEnv()
        obs, info = env.reset()
        
        assert obs is not None
        assert isinstance(info, dict)
    
    def test_env_step(self):
        """Test environment step."""
        env = PaperPlaneEnv()
        env.reset()
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)


class TestDDPGAgent:
    """Test DDPG RL agent."""
    
    @pytest.fixture
    def agent_and_env(self):
        """Fixture: agent and environment."""
        env = PaperPlaneEnv()
        env.reset()
        
        node_feature_dim = env.node_feature_dim
        action_dim = env.action_space.shape[0]
        state_vector_dim = env.state_vector_dim + env.action_dim
        
        agent = DDPGAgent(node_feature_dim, action_dim, state_vector_dim)
        
        return agent, env
    
    def test_agent_creation(self, agent_and_env):
        """Test agent initialization."""
        agent, env = agent_and_env
        
        assert agent.actor is not None
        assert agent.critic is not None
    
    def test_agent_forward(self, agent_and_env):
        """Test agent forward pass."""
        agent, env = agent_and_env
        
        obs, _ = env.reset()
        action = agent.select_action(obs, training=False)
        
        assert action.shape == env.action_space.shape
        assert np.all(action >= -1) and np.all(action <= 1)


class TestMultiFidelity:
    """Test multi-fidelity evaluation cascade."""
    
    @pytest.fixture
    def setup(self):
        """Fixture: setup multi-fidelity test."""
        action = np.array([0.25, 0.25, 0.75, 0.25, 0.5, 0.3, 0.3, 0.7, 0.7, 0.3, 0.5])
        mesh = fold_sheet(action, resolution=30)
        config = load_config()
        state = {
            'angle_of_attack_deg': config['goals']['angle_of_attack_deg'],
            'throw_speed_mps': config['goals']['throw_speed_mps'],
            'air_density_kgm3': config['environment']['air_density_kgm3'],
            'air_viscosity_pas': config['environment']['air_viscosity_pas']
        }
        gnn_model = GNNAeroSurrogate(node_feature_dim=7, hidden_dim=64)
        return mesh, state, gnn_model
    
    def test_multi_fidelity_cascade(self, setup):
        """Test multi-fidelity evaluation."""
        mesh, state, gnn_model = setup
        
        # Stage 1: GNN surrogate
        gnn_result = gnn_surrogate_cfd(mesh, state, gnn_model)
        
        # Stage 2: Classical surrogate (fallback)
        classical_result = surrogate_cfd(mesh, state)
        
        # Verify both produce reasonable results
        assert 0 <= gnn_result['range_est'] <= 100
        assert 0 <= classical_result['range_est'] <= 100


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_end_to_end_evaluation(self):
        """Test complete evaluation pipeline."""
        # 1. Create mesh
        action = np.random.uniform(0, 1, 25)
        mesh = fold_sheet(action, resolution=30)
        assert mesh is not None
        
        # 2. Get state
        config = load_config()
        state = {
            'angle_of_attack_deg': config['goals']['angle_of_attack_deg'],
            'throw_speed_mps': config['goals']['throw_speed_mps'],
            'air_density_kgm3': config['environment']['air_density_kgm3'],
            'air_viscosity_pas': config['environment']['air_viscosity_pas']
        }
        
        # 3. Evaluate with classical surrogate
        classical_result = surrogate_cfd(mesh, state)
        assert classical_result is not None
        
        # 4. Evaluate with GNN surrogate
        gnn_model = GNNAeroSurrogate(node_feature_dim=7, hidden_dim=64)
        gnn_result = gnn_surrogate_cfd(mesh, state, gnn_model)
        assert gnn_result is not None
        
        # 5. Verify consistency
        assert classical_result['range_est'] > 0
        assert gnn_result['range_est'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
