from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from torch_geometric.nn import HeteroConv, GCNConv, GATConv, GlobalAttention, SAGEConv
import torch.nn.functional as F

class BipartiteGNNExtractorV4(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        if not hasattr(observation_space, "spaces"):
            raise ValueError(
                "GNNWithPrefsExtractor expects observation_space to be gym.spaces.Dict"
            )

        self.pref_dim = observation_space.spaces["prefs"].shape[0]

        alloc_shape = observation_space.spaces["allocation_matrix"].shape  # (n_demands+1, n_resources)
        self.n_demands = alloc_shape[0] - 1
        self.n_resources = alloc_shape[1]

        # Feature dims for each node type (no padding)
        self.demand_feat_dim = 1 + self.n_resources + self.pref_dim  # production + allocation_row + prefs
        self.resource_feat_dim = 1 + self.n_demands + self.pref_dim  # available + allocation_col + prefs
        self.unallocated_feat_dim = 1 + self.n_resources + self.pref_dim  # 0.0 + unallocated + prefs

        # GNN layers for each edge type
        self.hetero_conv = HeteroConv({
            ("demand", "demand_to_resource", "resource"): SAGEConv((self.demand_feat_dim, self.resource_feat_dim), 64),
            ("resource", "resource_to_demand", "demand"): SAGEConv((self.resource_feat_dim, self.demand_feat_dim), 64),
            ("unallocated", "unallocated_to_resource", "resource"): SAGEConv((self.unallocated_feat_dim, self.resource_feat_dim), 64),
            ("resource", "resource_to_unallocated", "unallocated"): SAGEConv((self.resource_feat_dim, self.unallocated_feat_dim), 64),
        }, aggr="sum")

        self.norm = nn.ModuleDict({
            "demand": nn.LayerNorm(64),
            "resource": nn.LayerNorm(64),
            "unallocated": nn.LayerNorm(64),
        })

        # Attention pooling for each node type
        self.att_gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        self.att_pool = GlobalAttention(self.att_gate)

        # Final MLP
        self.linear_graph = nn.Linear(64 * 3 + self.pref_dim, 128)  # 3 node types pooled + prefs
        self.final = nn.Linear(128, features_dim)


    def forward(self, obs):
        device = next(self.parameters()).device
        allocation_matrix_batch = obs["allocation_matrix"].to(device)  # (batch, n_demands+1, n_resources)
        requirements_matrix_batch = obs["requirements_matrix"].to(device)
        prefs = obs["prefs"].to(device)  # (batch, pref_dim)

        batch_size, n_demands_plus1, n_resources = allocation_matrix_batch.shape
        n_demands = self.n_demands
        n_resources = self.n_resources

        pooled_all = []
        for i in range(batch_size):
            # --- Node features ---
            # Demand nodes (vectorized)
            if n_demands > 0:
                allocation_rows = allocation_matrix_batch[i, :n_demands, :]  # (n_demands, n_resources)
                productions = torch.min(allocation_rows, dim=1)[0].unsqueeze(1)  # (n_demands, 1)
                prefs_expanded = prefs[i].unsqueeze(0).expand(n_demands, -1)  # (n_demands, pref_dim)
                demand_feats = torch.cat([productions, allocation_rows, prefs_expanded], dim=1)  # (n_demands, demand_feat_dim)
            else:
                demand_feats = torch.empty(0, self.demand_feat_dim, device=device, dtype=torch.float)

            # Resource nodes (vectorized)
            available = allocation_matrix_batch[i, n_demands, :].unsqueeze(1)  # (n_resources, 1)
            allocation_cols = allocation_matrix_batch[i, :n_demands, :].T  # (n_resources, n_demands)
            prefs_expanded = prefs[i].unsqueeze(0).expand(n_resources, -1)  # (n_resources, pref_dim)
            resource_feats = torch.cat([available, allocation_cols, prefs_expanded], dim=1)  # (n_resources, resource_feat_dim)

            # Unallocated node
            unallocated = allocation_matrix_batch[i, n_demands, :].unsqueeze(0)  # (1, n_resources)
            zero_prod = torch.zeros(1, 1, device=device, dtype=torch.float)
            prefs_single = prefs[i].unsqueeze(0)  # (1, pref_dim)
            unallocated_feats = torch.cat([zero_prod, unallocated, prefs_single], dim=1)  # (1, unallocated_feat_dim)

            # --- Edge indices ---
            # demand_to_resource: demand -> resource (requirements)
            requirements_matrix = requirements_matrix_batch[i].tolist()
            demand_to_resource = [[], []]
            resource_to_demand = [[], []]
            for d in range(n_demands):
                for r in range(n_resources):
                    if requirements_matrix[d][r] == 1:
                        demand_to_resource[0].append(d)
                        demand_to_resource[1].append(r)
                        resource_to_demand[0].append(r)
                        resource_to_demand[1].append(d)

            # unallocated_to_resource: unallocated (0) -> resource
            unallocated_to_resource = [[], []]
            resource_to_unallocated = [[], []]
            for r in range(n_resources):
                unallocated_to_resource[0].append(0)
                unallocated_to_resource[1].append(r)
                resource_to_unallocated[0].append(r)
                resource_to_unallocated[1].append(0)

            # --- Build HeteroData dict ---
            data = {
                "demand": {"x": demand_feats},
                "resource": {"x": resource_feats},
                "unallocated": {"x": unallocated_feats},
                ("demand", "demand_to_resource", "resource"): {"edge_index": torch.tensor(demand_to_resource, dtype=torch.long, device=device) if demand_to_resource[0] else torch.empty(2, 0, dtype=torch.long, device=device)},
                ("resource", "resource_to_demand", "demand"): {"edge_index": torch.tensor(resource_to_demand, dtype=torch.long, device=device) if resource_to_demand[0] else torch.empty(2, 0, dtype=torch.long, device=device)},
                ("unallocated", "unallocated_to_resource", "resource"): {"edge_index": torch.tensor(unallocated_to_resource, dtype=torch.long, device=device)},
                ("resource", "resource_to_unallocated", "unallocated"): {"edge_index": torch.tensor(resource_to_unallocated, dtype=torch.long, device=device)},
            }

            # --- Forward pass ---
            x_dict = {k: v["x"] for k, v in data.items() if isinstance(k, str)}
            edge_index_dict = {k: v["edge_index"] for k, v in data.items() if isinstance(k, tuple)}
            x_dict = self.hetero_conv(x_dict, edge_index_dict)
            # LayerNorm + SiLU for each node type
            for ntype in x_dict:
                x_dict[ntype] = self.norm[ntype](x_dict[ntype])
                x_dict[ntype] = F.silu(x_dict[ntype])

            # Attention pooling for each node type
            pooled = []
            for ntype in ["demand", "resource", "unallocated"]:
                if x_dict[ntype].size(0) > 0:  # Check if there are any nodes of this type
                    pooled_ntype = self.att_pool(x_dict[ntype], batch=torch.zeros(x_dict[ntype].size(0), dtype=torch.long, device=device))
                    # GlobalAttention returns [1, 64], we need [64]
                    pooled_ntype = pooled_ntype.squeeze(0)  # Remove the first dimension
                else:
                    # If no nodes of this type, create a zero vector
                    pooled_ntype = torch.zeros(64, device=device, dtype=torch.float)
                pooled.append(pooled_ntype)
            # Concatenate all pooled node type embeddings for this batch item
            pooled_all.append(torch.cat(pooled, dim=0))  # Changed back to dim=0 since now they are 1D

        # Stack all batch items: (batch_size, total_pooled_dim)
        pooled_all = torch.stack(pooled_all, dim=0)
        # Concatenate with preferences: both should be 2D tensors
        x = torch.cat([pooled_all, prefs], dim=-1)
        x = self.linear_graph(x)
        out = self.final(x)
        return out