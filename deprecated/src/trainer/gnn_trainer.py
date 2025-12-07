"""
Graph Neural Network (GNN) trainer for ARC-style recursive learning.
Implements recursive pattern extraction similar to TRM paper methods for ARC intelligence tests.
Optimized for parallel GPU processing.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import yaml
from tqdm import tqdm
import time


class RecursiveGNNBlock(nn.Module):
    """
    Single recursive GNN block for hierarchical pattern learning.
    Processes graph structure recursively with attention mechanism.
    """
    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 4):
        super().__init__()
        self.gat = GATConv(in_channels, out_channels // num_heads, heads=num_heads, concat=True)
        self.graph_conv = GraphConv(out_channels, out_channels)
        self.ln1 = nn.LayerNorm(out_channels)
        self.ln2 = nn.LayerNorm(out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels * 2, out_channels)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Attention-based graph convolution
        x_attn = self.gat(x, edge_index)
        x = self.ln1(x_attn + x)
        
        # Standard graph convolution
        x_conv = self.graph_conv(x, edge_index)
        x = self.ln2(x_conv + x)
        
        # MLP refinement
        x_mlp = self.mlp(x)
        return x + x_mlp


class RecursiveGNNModel(nn.Module):
    """
    Multi-level recursive GNN for ARC-style pattern recognition.
    Inspired by TRM paper hierarchical learning methods.
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 64, 
                 output_dim: int = 1,
                 num_recursive_levels: int = 3,
                 num_heads: int = 4):
        super().__init__()
        
        self.num_levels = num_recursive_levels
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Build recursive levels
        self.recursive_blocks = nn.ModuleList([
            RecursiveGNNBlock(hidden_dim if i == 0 else hidden_dim, hidden_dim, num_heads)
            for i in range(num_recursive_levels)
        ])
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """Forward pass through recursive GNN levels."""
        # Project input
        x = self.input_projection(x)
        
        # Process through recursive levels
        for block in self.recursive_blocks:
            x = block(x, edge_index)
        
        # Global pooling for graph-level representation
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Output projection
        return self.output_projection(x)


class RecursiveGNNTrainer:
    """
    Trainer for recursive GNN model with GPU acceleration.
    Implements ARC-style pattern learning for optimization.
    """
    
    def __init__(self, 
                 model: RecursiveGNNModel,
                 device: torch.device,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = nn.MSELoss()
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
    def create_graph_from_folding(self, 
                                   folding_action: np.ndarray,
                                   n_folds: int) -> Data:
        """
        Convert folding action to graph representation.
        Creates nodes for each fold and edges based on spatial relationships.
        """
        # Number of nodes = folds + boundary nodes
        n_nodes = n_folds + 4  # 4 boundary nodes (corners)
        
        # Create node features from folding parameters
        # Each fold has 5 parameters: x1, y1, x2, y2, angle
        node_features = []
        for i in range(n_folds):
            fold_params = folding_action[i*5:(i+1)*5]
            node_features.append(fold_params)
        
        # Add boundary node features (fixed)
        boundary_features = np.array([
            [0, 0, 0, 0, 0],  # Top-left
            [1, 0, 0, 0, 0],  # Top-right
            [1, 1, 0, 0, 0],  # Bottom-right
            [0, 1, 0, 0, 0]   # Bottom-left
        ])
        
        node_features.extend(boundary_features)
        x = torch.tensor(np.array(node_features), dtype=torch.float32)
        
        # Create edges based on spatial proximity (recursive structure)
        edges = []
        for i in range(n_folds):
            # Connect to neighboring folds
            if i > 0:
                edges.append([i-1, i])
                edges.append([i, i-1])
            # Connect to boundary nodes (spatial hierarchy)
            for b in range(4):
                edges.append([i, n_folds + b])
                edges.append([n_folds + b, i])
        
        # Add boundary node connections
        for i in range(4):
            edges.append([n_folds + i, n_folds + (i+1)%4])
            edges.append([n_folds + (i+1)%4, n_folds + i])
        
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int, total_epochs: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
        
        for batch in progress_bar:
            batch = batch.to(self.device)
            
            # Forward pass
            output = self.model(batch.x, batch.edge_index, batch.batch)
            
            # Compute loss (predict normalized aerodynamic efficiency)
            target = batch.y.view_as(output)
            loss = self.criterion(output, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        self.training_history['loss'].append(avg_loss)
        self.training_history['learning_rate'].append(self.scheduler.get_last_lr()[0])
        self.scheduler.step()
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        for batch in val_loader:
            batch = batch.to(self.device)
            output = self.model(batch.x, batch.edge_index, batch.batch)
            target = batch.y.view_as(output)
            loss = self.criterion(output, target)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.training_history['val_loss'].append(avg_loss)
        
        return avg_loss
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              epochs: int = 50,
              early_stopping_patience: int = 10,
              progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Train the recursive GNN model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            progress_callback: Callback for progress updates
            
        Returns:
            Dictionary with training history and final model performance
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch, epochs)
            
            # Validate
            if val_loader:
                val_loss = self.validate(val_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                status = f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            else:
                status = f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}"
            
            # Progress callback
            if progress_callback:
                progress_callback(epoch + 1, epochs, train_loss, val_loss if val_loader else None)
            
            print(status)
        
        return {
            'train_loss': self.training_history['loss'],
            'val_loss': self.training_history['val_loss'],
            'learning_rate': self.training_history['learning_rate'],
            'best_val_loss': best_val_loss
        }


def create_synthetic_dataset(n_samples: int, n_folds: int, device: torch.device) -> DataLoader:
    """Create synthetic dataset of folding patterns and aerodynamic performance."""
    graphs = []
    
    for _ in range(n_samples):
        # Random folding action
        folding_action = np.random.uniform(0, 1, n_folds * 5)
        
        # Synthetic performance metric (CL/CD ratio as proxy for efficiency)
        aoa = np.random.uniform(0, 30)
        speed = np.random.uniform(5, 25)
        efficiency = 0.5 + 0.1 * (aoa / 30) + 0.02 * (speed / 25) + np.random.normal(0, 0.1)
        efficiency = np.clip(efficiency, 0.1, 2.0)  # Realistic range
        
        # Create graph
        n_nodes = n_folds + 4
        node_features = []
        for i in range(n_folds):
            fold_params = folding_action[i*5:(i+1)*5]
            node_features.append(fold_params)
        
        boundary_features = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ])
        node_features.extend(boundary_features)
        x = torch.tensor(np.array(node_features), dtype=torch.float32, device=device)
        
        # Create edges
        edges = []
        for i in range(n_folds):
            if i > 0:
                edges.append([i-1, i])
                edges.append([i, i-1])
            for b in range(4):
                edges.append([i, n_folds + b])
                edges.append([n_folds + b, i])
        
        for i in range(4):
            edges.append([n_folds + i, n_folds + (i+1)%4])
            edges.append([n_folds + (i+1)%4, n_folds + i])
        
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long, device=device)
        y = torch.tensor([[efficiency]], dtype=torch.float32, device=device)
        
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    
    return DataLoader(graphs, batch_size=32, shuffle=True)
