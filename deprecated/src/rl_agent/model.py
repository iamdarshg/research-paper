"""Custom PyTorch DDPG agent for paper plane folding with GPU optimization."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
import torch.cuda.amp as amp # New import for mixed precision

# For Graph Neural Networks
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data # Import Data for graph representation

from src.rl_agent.env import PaperPlaneEnv
from src.folding.sheet import load_config
from tqdm import tqdm

# Ensure PyTorch Geometric uses the correct device
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:
    """
    A replay buffer to store experiences for DDPG agent training,
    now handling graph data and state vectors.
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=int(capacity))
    
    def push(self, graph_data: Data, state_vector: np.ndarray, action: np.ndarray, reward: float, 
             next_graph_data: Data, next_state_vector: np.ndarray, done: bool):
        """Adds a new experience tuple to the buffer."""
        self.buffer.append((graph_data, state_vector, action, reward, next_graph_data, next_state_vector, done))
    
    def sample(self, batch_size: int):
        """
        Samples a batch of experiences from the buffer.
        
        Args:
            batch_size (int): The number of experiences to sample.
            
        Returns:
            Tuple[List[Data], np.ndarray, np.ndarray, np.ndarray, List[Data], np.ndarray, np.ndarray]: 
            A tuple containing batched graph data, state vectors, actions, rewards, next graph data, next state vectors, and done flags.
        """
        batch = random.sample(self.buffer, batch_size)
        graph_data_list, state_vector_list, action_list, reward_list, next_graph_data_list, next_state_vector_list, done_list = zip(*batch)
        
        # Convert to PyTorch Geometric DataBatch for efficient processing
        from torch_geometric.data import Batch
        batched_graph_data = Batch.from_data_list(list(graph_data_list))
        batched_next_graph_data = Batch.from_data_list(list(next_graph_data_list))

        state_vectors = torch.FloatTensor(np.array(state_vector_list))
        actions = torch.FloatTensor(np.array(action_list))
        rewards = torch.FloatTensor(np.array(reward_list)).unsqueeze(1)
        next_state_vectors = torch.FloatTensor(np.array(next_state_vector_list))
        dones = torch.FloatTensor(np.array(done_list)).unsqueeze(1)
        
        return batched_graph_data, state_vectors, actions, rewards, batched_next_graph_data, next_state_vectors, dones
    
    def __len__(self):
        return len(self.buffer)

# CONCEPTUAL: Actor network incorporating 3D spatially aware (e.g., GNN) features
class Actor(nn.Module):
    def __init__(self, node_feature_dim, action_dim, state_vector_dim, max_action=1.0):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # GNN layers for processing graph structure of the paper plane
        self.conv1 = GCNConv(node_feature_dim, 128)
        self.conv2 = GCNConv(128, 128)
        
        # Global pooling layer to get a fixed-size representation from the graph
        self.pool = global_mean_pool
        
        # MLP to output actions from the pooled graph representation combined with state vector
        self.fc_gnn_out = nn.Linear(128 + state_vector_dim, 256)
        self.fc_final = nn.Linear(256, action_dim)
        self.max_action = max_action
    
    def forward(self, graph_data: Data, state_vector: torch.Tensor):
        graph_data.x = graph_data.x.to(self.device)
        graph_data.edge_index = graph_data.edge_index.to(self.device)
        state_vector = state_vector.to(self.device)
        
        x = F.relu(self.conv1(graph_data.x, graph_data.edge_index))
        x = F.relu(self.conv2(x, graph_data.edge_index))
        
        # Global pooling to get a graph-level embedding
        # torch_geometric.data.Data automatically handles batch for DataBatch
        x_pooled = self.pool(x, graph_data.batch) 
        
        # Concatenate pooled graph features with the additional state vector
        x_combined = torch.cat([x_pooled, state_vector], dim=1)
        
        x = F.relu(self.fc_gnn_out(x_combined))
        x = torch.tanh(self.fc_final(x))
        
        return x * self.max_action

class Critic(nn.Module):
    def __init__(self, node_feature_dim, action_dim, state_vector_dim):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # GNN layers for state processing
        self.conv1_state = GCNConv(node_feature_dim, 128)
        self.conv2_state = GCNConv(128, 128)
        self.pool_state = global_mean_pool
        
        # MLP for value prediction, taking pooled state and action
        self.fc_critic_in = nn.Linear(128 + state_vector_dim + action_dim, 256) # Concatenate pooled state, state_vector and action
        self.fc_critic2 = nn.Linear(256, 256)
        self.fc_critic_out = nn.Linear(256, 1)
    
    def forward(self, graph_data: Data, state_vector: torch.Tensor, action: torch.Tensor):
        graph_data.x = graph_data.x.to(self.device)
        graph_data.edge_index = graph_data.edge_index.to(self.device)
        state_vector = state_vector.to(self.device)
        action = action.to(self.device)
        
        x_state = F.relu(self.conv1_state(graph_data.x, graph_data.edge_index))
        x_state = F.relu(self.conv2_state(x_state, graph_data.edge_index))
        x_state_pooled = self.pool_state(x_state, graph_data.batch)
        
        # Concatenate pooled state with action and the additional state vector
        x = torch.cat([x_state_pooled, state_vector, action], dim=1)
        x = F.relu(self.fc_critic_in(x))
        x = F.relu(self.fc_critic2(x))
        return self.fc_critic_out(x)

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) agent.
    
    Attributes:
        actor (Actor): The actor network.
        actor_target (Actor): The target actor network.
        critic (Critic): The critic network.
        critic_target (Critic): The target critic network.
        actor_optimizer (optim.Optimizer): Optimizer for the actor network.
        critic_optimizer (optim.Optimizer): Optimizer for the critic network.
        replay_buffer (ReplayBuffer): Buffer to store and sample experiences.
        tau (float): Soft update factor for target networks.
        batch_size (int): Batch size for training.
        gamma (float): Discount factor.
        action_dim (int): Dimension of the action space.
        state_vector_dim (int): Dimension of the non-graph state vector.
        node_feature_dim (int): Dimension of node features in graph.
        scaler (amp.GradScaler): Gradient scaler for mixed precision training.
        device (torch.device): The device (CPU or CUDA) the agent is running on.
        episode_count (int): Counter for the number of completed episodes.
    """
    def __init__(self, node_feature_dim: int, action_dim: int, state_vector_dim: int):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Training DDPGAgent on {self.device}")

        self.actor = Actor(node_feature_dim, action_dim, state_vector_dim).to(self.device)
        self.actor_target = Actor(node_feature_dim, action_dim, state_vector_dim).to(self.device)
        self.critic = Critic(node_feature_dim, action_dim, state_vector_dim).to(self.device)
        self.critic_target = Critic(node_feature_dim, action_dim, state_vector_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.replay_buffer = ReplayBuffer(int(1e6)) # Cast capacity to int
        self.tau = 0.005
        self.batch_size = 32
        self.gamma = 0.99
        self.action_dim = action_dim # Store action_dim
        self.state_vector_dim = state_vector_dim
        self.node_feature_dim = node_feature_dim
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Initialize GradScaler for mixed precision training
        self.scaler = amp.GradScaler(enabled=(self.device == 'cuda'))
        
        self.episode_count = 0 # Initialize episode_count
    
    def select_action(self, graph_data: Data, state_vector: np.ndarray, noise=0.1):
        graph_data = graph_data.to(self.device)
        state_vector_t = torch.FloatTensor(state_vector.reshape(1, -1)).to(self.device)
        with amp.autocast(enabled=(self.device == 'cuda')):
            action = self.actor(graph_data, state_vector_t)
        return action.cpu().data.numpy().flatten() + noise * np.random.randn(self.action_dim)
    
    def update(self):
        """
        Performs one step of DDPG training, updating actor and critic networks.
        Samples from replay buffer, computes losses, and updates network weights.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        graph_data_batch, state_vectors, actions, rewards, \
            next_graph_data_batch, next_state_vectors, dones = self.replay_buffer.sample(self.batch_size)
        
        # Move all sampled tensors to the correct device
        state_vectors = state_vectors.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_state_vectors = next_state_vectors.to(self.device)
        dones = dones.to(self.device)
        
        with amp.autocast(enabled=(self.device == 'cuda')):
            # Critic update
            next_actions = self.actor_target(next_graph_data_batch, next_state_vectors)
            target_q = rewards + self.gamma * self.critic_target(next_graph_data_batch, next_state_vectors, next_actions) * (1 - dones)
            current_q = self.critic(graph_data_batch, state_vectors, actions)
            critic_loss = F.mse_loss(current_q, target_q.detach())
        
        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_optimizer)
        
        with amp.autocast(enabled=(self.device == 'cuda')):
            actor_loss = -self.critic(graph_data_batch, state_vectors, self.actor(graph_data_batch, state_vectors)).mean()
        
        self.actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_optimizer)
        self.scaler.update() # Update the scaler for the next iteration

        # Target network update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self, env, total_episodes=100, progress_callback=None, mesh_callback=None):
        # The env.reset() now returns (graph_data, state_vector), info
        # The initial `state` in the DDPGAgent.train loop now consists of both graph_data and state_vector_np
        initial_obs, info = env.reset()
        graph_data: Data = initial_obs['graph']
        state_vector_np: np.ndarray = initial_obs['vector']

        episode_reward = 0
        episode_max_range = 0
        rewards = []
        ranges = []
        print(f"Starting training for {total_episodes} episodes...")
        
        pbar = tqdm(total=total_episodes, desc='Training', unit='episode')
        
        current_step_in_episode = 0

        while self.episode_count < total_episodes:
            action = self.select_action(graph_data, state_vector_np)
            
            # env.step() will now return (next_graph_data, next_state_vector_np), reward, terminated, truncated, info
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_graph_data: Data = next_obs['graph']
            next_state_vector_np: np.ndarray = next_obs['vector']
            done = terminated or truncated
            
            current_mesh = info.get('mesh', None)

            self.replay_buffer.push(graph_data, state_vector_np, action, reward, next_graph_data, next_state_vector_np, done)
            
            self.update()

            graph_data = next_graph_data
            state_vector_np = next_state_vector_np
            episode_reward += reward
            episode_max_range = max(episode_max_range, info['range'])
            current_step_in_episode += 1

            if done:
                rewards.append(episode_reward)
                ranges.append(episode_max_range)
                initial_obs, _ = env.reset()
                graph_data = initial_obs['graph']
                state_vector_np = initial_obs['vector']
                episode_reward = 0
                episode_max_range = 0
                current_step_in_episode = 0
                self.episode_count += 1
                pbar.update(1)

            if progress_callback:
                progress_callback(
                    current_episode=self.episode_count,
                    total_episodes=total_episodes,
                    ep_reward=episode_reward,
                    ep_range=episode_max_range,
                    avg_range_10=np.mean(ranges[-10:]) if len(ranges) >= 10 else np.mean(ranges) if len(ranges) > 0 else 0.0
                )
            
            if mesh_callback and current_mesh is not None and (done or current_step_in_episode % 20 == 0):
                mesh_callback(current_mesh)

            pbar.set_postfix({
                'episode': self.episode_count,
                'ep_reward': f'{episode_reward:.2f}',
                'ep_range': f'{episode_max_range:.2f}m',
                'avg_range_10': f'{np.mean(ranges[-10:]):.2f}m' if len(ranges) >= 10 else f'{np.mean(ranges) if len(ranges) > 0 else 0.0:.2f}m'
            })
        
        pbar.close()
        return self, rewards, ranges
