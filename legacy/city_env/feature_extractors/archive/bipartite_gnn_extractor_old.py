from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv

class BipartiteGNNWithPrefsExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        if not hasattr(observation_space, "spaces"):
            raise ValueError(
                "GNNWithPrefsExtractor expects observation_space to be gym.spaces.Dict"
            )

        self.pref_dim = observation_space.spaces["prefs"].shape[0]

        # Enriched node features: [is_demand, is_resource, production/available, allocation_row/col, prefs]
        # Compute in_features based on observation_space
        # n_demands, n_resources are not directly available, so we infer from observation_space
        alloc_shape = observation_space.spaces["allocation_matrix"].shape  # (n_demands+1, n_resources)
        n_demands = alloc_shape[0] - 1
        n_resources = alloc_shape[1]
        # Demand node: [1.0, 0.0, production] + allocation_row + prefs
        # Unallocated node: [1.0, 0.0, 0.0] + unallocated + prefs
        # Resource node: [0.0, 1.0, available] + allocation_col + prefs
        # All have same length: 2 + 1 + n_resources + pref_dim
        in_features = 2 + 1 + n_resources + self.pref_dim
        self.conv1 = GCNConv(in_features, 64)
        self.norm1 = nn.LayerNorm(64)
        self.conv2 = GCNConv(64, 64)
        self.norm2 = nn.LayerNorm(64)
        self.conv3 = GATConv(64, 64)
        self.norm3 = nn.LayerNorm(64)
        self.linear_graph = nn.Linear(64 * 2, 128)
        self.final = nn.Linear(128 + self.pref_dim, features_dim)


    def forward(self, obs):
        device = next(self.parameters()).device
        allocation_matrix_batch = obs["allocation_matrix"].to(device)  # (batch, n_demands+1, n_resources)
        requirements_matrix_batch = obs["requirements_matrix"].to(device)
        prefs = obs["prefs"].to(device)  # (batch, pref_dim)

        batch_size, n_demands_plus1, n_resources = allocation_matrix_batch.shape
        n_demands = n_demands_plus1 - 1
        num_nodes = n_demands + 1 + n_resources

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
                    batched_edge_weights.append(1.0)
                else:
                    batched_edge_weights.append(0.0)

            # Shift node indices for batch
            edge_list_shifted = [[u + i * num_nodes, v + i * num_nodes] for u, v in edge_list]
            batched_edge_list.extend(edge_list_shifted)

            # Enriched node features
            node_features = []
            # Demand nodes
            for d in range(n_demands):
                allocation_row = allocation_matrix_batch[i, d, :].tolist()
                production = float(min(allocation_row)) if allocation_row else 0.0
                node_features.append(
                    [1.0, 0.0, production] + allocation_row + prefs[i].tolist()
                )
            # Unallocated node
            unallocated = allocation_matrix_batch[i, n_demands, :].tolist()
            node_features.append([1.0, 0.0, 0.0] + unallocated + prefs[i].tolist())
            # Resource nodes
            for r in range(n_resources):
                available = allocation_matrix_batch[i, n_demands, r].item()
                allocation_col = allocation_matrix_batch[i, :n_demands, r].tolist()
                # Pad allocation_col to n_resources
                if len(allocation_col) < n_resources:
                    allocation_col += [0.0] * (n_resources - len(allocation_col))
                node_features.append(
                    [0.0, 1.0, available] + allocation_col + prefs[i].tolist()
                )
            batched_node_features.extend(node_features)

        # All GNN layers are initialized in __init__

        edge_index_batched = torch.tensor(batched_edge_list, dtype=torch.long, device=device).T
        # Set all edge weights to 1 (unweighted graph)
        edge_weights_batched = torch.ones(len(batched_edge_list), dtype=torch.float, device=device)
        x_batched = torch.tensor(batched_node_features, dtype=torch.float, device=device)

        batch_vector = torch.arange(batch_size, device=x_batched.device).repeat_interleave(num_nodes)

        x = self.conv1(x_batched, edge_index_batched, edge_weights_batched)
        x = self.norm1(x)
        x = torch.relu(x)
        x = self.conv2(x, edge_index_batched, edge_weights_batched)
        x = self.norm2(x)
        x = torch.relu(x)
        x = self.conv3(x, edge_index_batched)
        x = self.norm3(x)
        x = torch.relu(x)

        demand_indices = []
        resource_indices = []
        for i in range(batch_size):
            offset = i * num_nodes
            demand_indices.extend([offset + j for j in range(n_demands + 1)])
            resource_indices.extend([offset + n_demands + 1 + j for j in range(n_resources)])

        demand_indices = torch.tensor(demand_indices, device=x.device)
        resource_indices = torch.tensor(resource_indices, device=x.device)

        x_demand = global_mean_pool(x[demand_indices], batch_vector[demand_indices])
        x_resource = global_mean_pool(x[resource_indices], batch_vector[resource_indices])
        x = torch.cat([x_demand, x_resource], dim=-1)

        x = self.linear_graph(x)
        combined = torch.cat([x, prefs], dim=-1)
        out = self.final(combined)
        return out