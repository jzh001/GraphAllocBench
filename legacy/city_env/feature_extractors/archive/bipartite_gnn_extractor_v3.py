from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention

class BipartiteGNNExtractorV3(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        if not hasattr(observation_space, "spaces"):
            raise ValueError(
                "GNNWithPrefsExtractor expects observation_space to be gym.spaces.Dict"
            )

        self.pref_dim = observation_space.spaces["prefs"].shape[0]

        alloc_shape = observation_space.spaces["allocation_matrix"].shape  # (n_demands+1, n_resources)
        n_demands = alloc_shape[0] - 1
        n_resources = alloc_shape[1]
        max_alloc = max(n_demands, n_resources)
        self.type_emb_dim = 4  # You can tune this dimension
        in_features = self.type_emb_dim + 1 + max_alloc + self.pref_dim
        self.type_embedding = nn.Embedding(3, self.type_emb_dim)  # 3 node types: demand, resource, unallocated
        self.conv1 = GCNConv(in_features, 64)
        self.norm1 = nn.LayerNorm(64)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = GCNConv(64, 64)
        self.norm2 = nn.LayerNorm(64)
        self.dropout2 = nn.Dropout(0.1)
        self.conv3 = GCNConv(64, 64)
        self.norm3 = nn.LayerNorm(64)
        self.dropout3 = nn.Dropout(0.1)

        # Attention gate: a small MLP
        self.att_gate = nn.Sequential(
            nn.Linear(192, 32),  # 64*3 for DenseNet-like concat
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        self.att_pool = GlobalAttention(self.att_gate)
        # DenseNet-like: concatenate x1, x2, x3 before pooling, so feature size is 64*3
        # Pooling will concatenate mean and attn, so final size is 64*3*2 = 384
        self.linear_graph = nn.Linear(384, 128)
        self.final = nn.Linear(128, features_dim)


    def forward(self, obs):
        device = next(self.parameters()).device
        allocation_matrix_batch = obs["allocation_matrix"].to(device)  # (batch, n_demands+1, n_resources)
        requirements_matrix_batch = obs["requirements_matrix"].to(device)
        prefs = obs["prefs"].to(device)  # (batch, pref_dim)

        batch_size, n_demands_plus1, n_resources = allocation_matrix_batch.shape
        n_demands = n_demands_plus1 - 1
        num_nodes = n_demands + 1 + n_resources
        max_alloc = max(n_demands, n_resources)

        batched_edge_list = []
        batched_edge_weights = []
        batched_node_features = []

        for i in range(batch_size):
            requirements_matrix = requirements_matrix_batch[i].tolist()
            edge_list = []
            for u in range(n_demands):
                for v in range(n_resources):
                    if requirements_matrix[u][v] == 1:
                        edge_list.append([u, v + n_demands + 1])
            for r in range(n_resources):
                edge_list.append([n_demands, r + n_demands + 1])
            edge_list = edge_list + [[dst, src] for src, dst in edge_list]

            # Edge weights
            for u, v in edge_list:
                if u <= n_demands and v > n_demands:
                    batched_edge_weights.append(allocation_matrix_batch[i, u, v - n_demands - 1].item())
                elif v <= n_demands and u > n_demands:
                    batched_edge_weights.append(allocation_matrix_batch[i, v, u - n_demands - 1].item())
                else:
                    batched_edge_weights.append(0.0)

            # Shift node indices for batch
            edge_list_shifted = [[u + i * num_nodes, v + i * num_nodes] for u, v in edge_list]
            batched_edge_list.extend(edge_list_shifted)

            # Enriched node features with learnable type embedding and early preference injection
            node_features = []
            # Demand nodes
            for d in range(n_demands):
                allocation_row = allocation_matrix_batch[i, d, :].tolist()
                production = float(min(allocation_row)) if allocation_row else 0.0
                if len(allocation_row) < max_alloc:
                    allocation_row += [0.0] * (max_alloc - len(allocation_row))
                type_emb = self.type_embedding(torch.tensor([0], device=device)).squeeze(0)  # 0: demand
                # Inject preferences into node features
                node_features.append(torch.cat([
                    type_emb,
                    torch.tensor([production] + allocation_row, device=device, dtype=torch.float),
                    prefs[i]
                ]))
            # Unallocated node
            unallocated = allocation_matrix_batch[i, n_demands, :].tolist()
            if len(unallocated) < max_alloc:
                unallocated += [0.0] * (max_alloc - len(unallocated))
            type_emb = self.type_embedding(torch.tensor([2], device=device)).squeeze(0)  # 2: unallocated
            node_features.append(torch.cat([
                type_emb,
                torch.tensor([0.0] + unallocated, device=device, dtype=torch.float),
                prefs[i]
            ]))
            # Resource nodes
            for r in range(n_resources):
                available = allocation_matrix_batch[i, n_demands, r].item()
                allocation_col = allocation_matrix_batch[i, :n_demands, r].tolist()
                if len(allocation_col) < max_alloc:
                    allocation_col += [0.0] * (max_alloc - len(allocation_col))
                type_emb = self.type_embedding(torch.tensor([1], device=device)).squeeze(0)  # 1: resource
                node_features.append(torch.cat([
                    type_emb,
                    torch.tensor([available] + allocation_col, device=device, dtype=torch.float),
                    prefs[i]
                ]))
            batched_node_features.extend(node_features)

        edge_index_batched = torch.tensor(batched_edge_list, dtype=torch.long, device=device).T
        edge_weights_batched = torch.tensor(batched_edge_weights, dtype=torch.float, device=device)
        x_batched = torch.stack(batched_node_features, dim=0)

        batch_vector = torch.arange(batch_size, device=x_batched.device).repeat_interleave(num_nodes)

        x1 = self.conv1(x_batched, edge_index_batched, edge_weights_batched)
        x1 = self.norm1(x1)
        x1 = F.silu(x1)
        x1 = self.dropout1(x1)

        x2 = self.conv2(x1, edge_index_batched, edge_weights_batched)
        x2 = x2 + x1  # skip connection
        x2 = self.norm2(x2)
        x2 = F.silu(x2)
        x2 = self.dropout2(x2)

        x3 = self.conv3(x2, edge_index_batched, edge_weights_batched)
        x3 = x3 + x2  # skip connection
        x3 = self.norm3(x3)
        x3 = F.silu(x3)
        x3 = self.dropout3(x3)

        # DenseNet-like: concatenate all layer outputs
        x_cat = torch.cat([x1, x2, x3], dim=-1)

        # Pooling: concatenate mean and attention pooling
        mean_pooled = global_mean_pool(x_cat, batch_vector)
        attn_pooled = self.att_pool(x_cat, batch_vector)
        pooled = torch.cat([mean_pooled, attn_pooled], dim=-1)

        x = self.linear_graph(pooled)
        out = self.final(x)
        return out