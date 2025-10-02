from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
import torch.nn.functional as F

class BipartiteGNNWithPrefsExtractor(BaseFeaturesExtractor):
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
        self.conv2 = GCNConv(64, 64)
        self.norm2 = nn.LayerNorm(64)
        self.conv3 = GCNConv(64, 64)
        self.norm3 = nn.LayerNorm(64)
        self.linear_graph = nn.Linear(64 * (n_demands + 1 + n_resources) + self.pref_dim, 128)
        # self.linear_graph = nn.Linear(64 * (n_demands + 1), 128)
        self.final = nn.Linear(128, features_dim)


    def forward(self, obs):
        device = next(self.parameters()).device
        allocation_matrix_batch = obs["allocation_matrix"].to(device)  # (batch, n_demands+1, n_resources)
        requirements_matrix_batch = obs["requirements_matrix"].to(device)
        prefs = obs["prefs"].to(device)  # (batch, pref_dim)

        batch_size, n_demands_plus1, n_resources = allocation_matrix_batch.shape
        n_demands = n_demands_plus1 - 1
        num_nodes = n_demands + 1 + n_resources

        # Ensure all node features have the same length by padding allocation vectors to max(n_demands, n_resources)
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

            # Enriched node features with learnable type embedding
            node_features = []
            # Demand nodes
            for d in range(n_demands):
                allocation_row = allocation_matrix_batch[i, d, :].tolist()
                production = float(min(allocation_row)) if allocation_row else 0.0
                if len(allocation_row) < max_alloc:
                    allocation_row += [0.0] * (max_alloc - len(allocation_row))
                type_emb = self.type_embedding(torch.tensor([0], device=device)).squeeze(0)  # 0: demand
                node_features.append(torch.cat([
                    type_emb,
                    torch.tensor([production] + allocation_row + prefs[i].tolist(), device=device, dtype=torch.float)
                ]))
            # Unallocated node
            unallocated = allocation_matrix_batch[i, n_demands, :].tolist()
            if len(unallocated) < max_alloc:
                unallocated += [0.0] * (max_alloc - len(unallocated))
            type_emb = self.type_embedding(torch.tensor([2], device=device)).squeeze(0)  # 2: unallocated
            node_features.append(torch.cat([
                type_emb,
                torch.tensor([0.0] + unallocated + prefs[i].tolist(), device=device, dtype=torch.float)
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
                    torch.tensor([available] + allocation_col + prefs[i].tolist(), device=device, dtype=torch.float)
                ]))
            batched_node_features.extend(node_features)

        # All GNN layers are initialized in __init__

        edge_index_batched = torch.tensor(batched_edge_list, dtype=torch.long, device=device).T
        # Set all edge weights to 1 (unweighted graph)
        # edge_weights_batched = torch.ones(len(batched_edge_list), dtype=torch.float, device=device)
        edge_weights_batched = torch.tensor(batched_edge_weights, dtype=torch.float, device=device)
        x_batched = torch.stack(batched_node_features, dim=0)

        batch_vector = torch.arange(batch_size, device=x_batched.device).repeat_interleave(num_nodes)

        x1 = self.conv1(x_batched, edge_index_batched, edge_weights_batched)
        x1 = self.norm1(x1)
        x1 = F.silu(x1)

        x2 = self.conv2(x1, edge_index_batched, edge_weights_batched)
        x2 = x2 + x1  # skip connection
        x2 = self.norm2(x2)
        x2 = F.silu(x2)

        x3 = self.conv3(x2, edge_index_batched, edge_weights_batched)
        x3 = x3 + x2  # skip connection
        x3 = self.norm3(x3)
        x3 = F.silu(x3)

        x = x3

        demand_indices = []
        resource_indices = []
        for i in range(batch_size):
            offset = i * num_nodes
            demand_indices.extend([offset + j for j in range(n_demands + 1)])
            resource_indices.extend([offset + n_demands + 1 + j for j in range(n_resources)])

        demand_indices = torch.tensor(demand_indices, device=x.device)
        resource_indices = torch.tensor(resource_indices, device=x.device)

        x_demand = x[demand_indices].reshape(batch_size, -1)
        x_resource = x[resource_indices].reshape(batch_size, -1)
        x = torch.cat([x_demand, x_resource], dim=-1)
        # Append preferences to the GNN output
        x = torch.cat([x, prefs], dim=-1)

        x = self.linear_graph(x)
        out = self.final(x)
        return out