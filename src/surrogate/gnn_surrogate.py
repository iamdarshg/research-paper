"""Graph Neural Network-based aerodynamic surrogate for paper airplane folding."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pickle

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GNNAeroSurrogate(nn.Module):
    """Graph Isomorphism Network (GIN) with edge updates for aerodynamic prediction."""

    def __init__(
        self,
        node_feature_dim: int = 7,
        edge_feature_dim: int = 4,
        global_feature_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 4,
        output_dim: int = 3,  # CL, CD, range
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.global_feature_dim = global_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Node embedding
        self.node_embed = nn.Linear(node_feature_dim, hidden_dim)

        # Message passing layers with edge updates
        self.mp_layers = nn.ModuleList()
        self.edge_mlps = nn.ModuleList()
        self.node_mlps = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_dim if i > 0 else hidden_dim
            self.mp_layers.append(GraphConv(in_dim, hidden_dim))
            
            # Edge update MLP: [h_i, h_j, e_ij]
            self.edge_mlps.append(nn.Sequential(
                nn.Linear(2 * hidden_dim + edge_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
            
            # Node update MLP: [h_i, aggregated_messages]
            self.node_mlps.append(nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))

        # Graph pooling and final prediction head
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim + global_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            data: torch_geometric.data.Data object with:
                - x: node features [num_nodes, node_feature_dim]
                - edge_index: edge connectivity [2, num_edges]
                - edge_attr: edge features [num_edges, edge_feature_dim]
                - u: global features [batch_size, global_feature_dim]
                - batch: batch assignment for each node [num_nodes]
        
        Returns:
            Predictions [batch_size, output_dim]
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        u = data.u if hasattr(data, 'u') else torch.zeros(data.num_graphs, self.global_feature_dim, device=x.device)
        batch = data.batch

        # Initial node embedding
        h = self.node_embed(x)

        # Message passing with edge updates
        for i in range(self.num_layers):
            # Message passing on graph
            h_agg = self.mp_layers[i](h, edge_index)

            # Edge update: compute edge messages based on source/target nodes
            src, dst = edge_index[0], edge_index[1]
            edge_input = torch.cat([h[src], h[dst], edge_attr], dim=1)
            edge_msg = self.edge_mlps[i](edge_input)

            # Aggregate messages for each node
            msg_agg = torch.zeros_like(h)
            msg_agg.index_add_(0, dst, edge_msg)

            # Node update
            node_input = torch.cat([h, msg_agg], dim=1)
            h = h + self.node_mlps[i](node_input)  # Residual connection

        # Graph-level pooling
        graph_rep = global_mean_pool(h, batch)

        # Concatenate with global features
        graph_feat = torch.cat([graph_rep, u], dim=1)

        # Predict outputs
        out = self.pool_mlp(graph_feat)

        return out


def mesh_to_graph(mesh: trimesh.Trimesh, state: Dict, device: str = 'cpu') -> Data:
    """
    Convert a trimesh to a PyTorch Geometric Data object.
    
    Args:
        mesh: trimesh.Trimesh object
        state: dict with keys like 'angle_of_attack_deg', 'throw_speed_mps', etc.
        device: 'cpu' or 'cuda'
    
    Returns:
        torch_geometric.data.Data object
    """
    vertices = mesh.vertices
    faces = mesh.faces
    normals = mesh.face_normals

    # Node features: [x, y, z, n_x, n_y, n_z, curvature]
    vertex_normals = np.zeros_like(vertices)
    vertex_curvature = np.zeros(len(vertices))
    
    # Compute vertex normals as average of adjacent face normals
    for v_idx in range(len(vertices)):
        adj_faces = np.where(np.any(faces == v_idx, axis=1))[0]
        if len(adj_faces) > 0:
            vertex_normals[v_idx] = normals[adj_faces].mean(axis=0)
            # Simple curvature proxy: variation in normals
            vertex_curvature[v_idx] = normals[adj_faces].std(axis=0).mean()

    node_features = np.hstack([
        vertices,  # x, y, z
        vertex_normals,  # n_x, n_y, n_z
        vertex_curvature[:, None]  # curvature
    ]).astype(np.float32)

    # Construct edge list from faces
    edges = set()
    for face in faces:
        edges.add((min(face[0], face[1]), max(face[0], face[1])))
        edges.add((min(face[1], face[2]), max(face[1], face[2])))
        edges.add((min(face[2], face[0]), max(face[2], face[0])))
    
    edges = np.array(list(edges), dtype=np.int64).T
    edge_index = torch.tensor(edges, dtype=torch.long, device=device)

    # Edge features: [dx, dy, dz, dihedral]
    edge_attr = []
    for i in range(edges.shape[1]):
        src, dst = edges[0, i], edges[1, i]
        pos_diff = vertices[dst] - vertices[src]
        dihedral = 0.0  # Placeholder; could compute actual dihedral angle
        edge_attr.append(np.concatenate([pos_diff, [dihedral]]))
    
    edge_attr = np.array(edge_attr, dtype=np.float32)

    # Global features: [aoa_rad, v_inf, rho, mu]
    aoa_deg = state.get('angle_of_attack_deg', 10.0)
    v_inf = state.get('throw_speed_mps', 10.0)
    rho = state.get('air_density_kgm3', 1.225)
    mu = state.get('air_viscosity_pas', 1.8e-5)
    
    global_features = np.array(
        [np.deg2rad(aoa_deg), v_inf, rho, mu],
        dtype=np.float32
    )

    # Create PyTorch Geometric Data object
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32, device=device),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32, device=device),
        u=torch.tensor(global_features, dtype=torch.float32, device=device).unsqueeze(0),
        batch=torch.zeros(len(vertices), dtype=torch.long, device=device)
    )

    return data


class GNNSurrogateTrainer:
    """Trainer for GNN aerodynamic surrogate."""

    def __init__(
        self,
        model: GNNAeroSurrogate,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': []}

    def train_step(self, data_list: List[Data], targets: torch.Tensor) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        predictions = []
        for data in data_list:
            pred = self.model(data)
            predictions.append(pred)
        predictions = torch.cat(predictions, dim=0)

        # Loss
        loss = self.criterion(predictions, targets.to(self.device))
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_step(self, data_list: List[Data], targets: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Evaluation step."""
        self.model.eval()
        with torch.no_grad():
            predictions = []
            for data in data_list:
                pred = self.model(data)
                predictions.append(pred)
            predictions = torch.cat(predictions, dim=0)

            loss = self.criterion(predictions, targets.to(self.device))

        return loss.item(), predictions

    def train(self, data_list: List[Data], targets: torch.Tensor, epochs: int = 100, val_split: float = 0.2):
        """Train the GNN model."""
        n_samples = len(data_list)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val

        train_data = data_list[:n_train]
        val_data = data_list[n_train:]
        train_targets = targets[:n_train]
        val_targets = targets[n_train:]

        for epoch in range(epochs):
            train_loss = self.train_step(train_data, train_targets)
            val_loss, _ = self.eval_step(val_data, val_targets)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    def save(self, path: str):
        """Save model to disk."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """Load model from disk."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def gnn_surrogate_cfd(mesh: trimesh.Trimesh, state: Dict, model: Optional[GNNAeroSurrogate] = None) -> Dict:
    """
    Predict aerodynamic coefficients using GNN surrogate.
    
    Args:
        mesh: trimesh.Trimesh folded airplane
        state: dict with environmental/control params
        model: Trained GNNAeroSurrogate model. If None, raises error.
    
    Returns:
        dict with 'cl', 'cd', 'range_est' keys
    """
    if model is None:
        raise ValueError("GNN model must be provided for inference")

    model.eval()
    device = next(model.parameters()).device

    # Convert mesh to graph
    data = mesh_to_graph(mesh, state, device=str(device).split(':')[0])

    # Inference
    with torch.no_grad():
        out = model(data)  # [1, 3] -> [cl, cd, range]
        cl, cd, range_est = out[0].cpu().numpy()

    return {
        'cl': float(np.clip(cl, -2, 2)),
        'cd': float(np.clip(cd, 0.01, 1)),
        'range_est': float(np.clip(range_est, 0, 100)),
        'features': {}  # Placeholder for compatibility
    }
