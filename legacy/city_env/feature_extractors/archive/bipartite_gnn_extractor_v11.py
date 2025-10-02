from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
from torch_geometric.data import HeteroData, Batch
import torch.nn.functional as F

class BipartiteGNNExtractorV11(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        if not hasattr(observation_space, "spaces"):
            raise ValueError("Requires gym.spaces.Dict")
        self.pref_dim = observation_space.spaces["prefs"].shape[0]
        alloc_shape = observation_space.spaces["allocation_matrix"].shape  # (n_demands+1, n_resources)
        self.n_demands = alloc_shape[0] - 1
        self.n_resources = alloc_shape[1]
        self.demand_feat_dim = 1 + self.n_resources + self.pref_dim
        self.resource_feat_dim = 1 + self.n_demands + self.pref_dim
        self.unallocated_feat_dim = 1 + self.n_resources + self.pref_dim


        # Two-layer SAGEConv for each edge type
        self.hetero_conv1 = HeteroConv({
            ("demand", "demand_to_resource", "resource"): SAGEConv((self.demand_feat_dim, self.resource_feat_dim), 64),
            ("resource", "resource_to_demand", "demand"): SAGEConv((self.resource_feat_dim, self.demand_feat_dim), 64),
            ("unallocated", "unallocated_to_resource", "resource"): SAGEConv((self.unallocated_feat_dim, self.resource_feat_dim), 64),
            ("resource", "resource_to_unallocated", "unallocated"): SAGEConv((self.resource_feat_dim, self.unallocated_feat_dim), 64),
        }, aggr="sum")
        self.hetero_conv2 = HeteroConv({
            ("demand", "demand_to_resource", "resource"): SAGEConv((64, 64), 64),
            ("resource", "resource_to_demand", "demand"): SAGEConv((64, 64), 64),
            ("unallocated", "unallocated_to_resource", "resource"): SAGEConv((64, 64), 64),
            ("resource", "resource_to_unallocated", "unallocated"): SAGEConv((64, 64), 64),
        }, aggr="sum")

        # LayerNorm for each node type after GNN
        self.norms = nn.ModuleDict({
            "demand": nn.LayerNorm(64),
            "resource": nn.LayerNorm(64),
            "unallocated": nn.LayerNorm(64),
        })

        self.final = nn.Sequential(
            nn.LayerNorm(64 * 3 + self.pref_dim),
            nn.ReLU(),
            nn.Linear(64 * 3 + self.pref_dim, features_dim)
        )

    def forward(self, obs):
        device = next(self.parameters()).device
        allocation_matrix_batch = obs["allocation_matrix"].to(device)
        requirements_matrix_batch = obs["requirements_matrix"].to(device)
        prefs = obs["prefs"].to(device)

        batch_size = allocation_matrix_batch.shape[0]
        n_demands, n_resources = self.n_demands, self.n_resources
        data_list = []
        for i in range(batch_size):
            if n_demands > 0:
                allocation_rows = allocation_matrix_batch[i, :n_demands, :]
                productions = torch.min(allocation_rows, dim=1)[0].unsqueeze(1)
                prefs_expanded = prefs[i].unsqueeze(0).expand(n_demands, -1)
                demand_feats = torch.cat([productions, allocation_rows, prefs_expanded], dim=1)
            else:
                demand_feats = torch.empty(0, self.demand_feat_dim, device=device)
            # resources
            available = allocation_matrix_batch[i, n_demands, :].unsqueeze(1)
            allocation_cols = allocation_matrix_batch[i, :n_demands, :].T
            prefs_expanded = prefs[i].unsqueeze(0).expand(n_resources, -1)
            resource_feats = torch.cat([available, allocation_cols, prefs_expanded], dim=1)
            # unallocated
            unallocated = allocation_matrix_batch[i, n_demands, :].unsqueeze(0)
            zero_prod = torch.zeros(1, 1, device=device)
            prefs_single = prefs[i].unsqueeze(0)
            unallocated_feats = torch.cat([zero_prod, unallocated, prefs_single], dim=1)
            # edge index construction as before (no changes)
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
            unallocated_to_resource = ([0] * n_resources, [r for r in range(n_resources)])
            resource_to_unallocated = ([r for r in range(n_resources)], [0] * n_resources)
            data = HeteroData()
            data['demand'].x = demand_feats
            data['resource'].x = resource_feats
            data['unallocated'].x = unallocated_feats
            data['demand', 'demand_to_resource', 'resource'].edge_index = (
                torch.tensor(demand_to_resource, dtype=torch.long, device=device)
                if demand_to_resource[0] else torch.empty(2, 0, dtype=torch.long, device=device)
            )
            data['resource', 'resource_to_demand', 'demand'].edge_index = (
                torch.tensor(resource_to_demand, dtype=torch.long, device=device)
                if resource_to_demand[0] else torch.empty(2, 0, dtype=torch.long, device=device)
            )
            data['unallocated', 'unallocated_to_resource', 'resource'].edge_index = torch.tensor(unallocated_to_resource, dtype=torch.long, device=device)
            data['resource', 'resource_to_unallocated', 'unallocated'].edge_index = torch.tensor(resource_to_unallocated, dtype=torch.long, device=device)
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        x_dict = batch.x_dict
        # First GATConv layer
        x_out1 = self.hetero_conv1(x_dict, batch.edge_index_dict)
        for ntype in ["demand", "resource", "unallocated"]:
            if x_out1[ntype].size(0) > 0:
                x_out1[ntype] = F.relu(x_out1[ntype])
            else:
                x_out1[ntype] = torch.zeros(0, 64, device=device)
        # Second GATConv layer
        x_out2 = self.hetero_conv2(x_out1, batch.edge_index_dict)
        # Apply LayerNorm after second layer
        for ntype in ["demand", "resource", "unallocated"]:
            if x_out2[ntype].size(0) > 0:
                x_out2[ntype] = self.norms[ntype](F.relu(x_out2[ntype]))
            else:
                x_out2[ntype] = torch.zeros(0, 64, device=device)
        # global mean pooling over all node types
        pooled = []
        for ntype in ["demand", "resource", "unallocated"]:
            if x_out2[ntype].size(0) > 0:
                pooled_ntype = global_mean_pool(x_out2[ntype], batch[ntype].batch)
            else:
                pooled_ntype = torch.zeros(batch_size, 64, device=device)
            pooled.append(pooled_ntype)
        pooled_all = torch.cat(pooled, dim=1)
        x = torch.cat([pooled_all, prefs], dim=-1)
        out = self.final(x)
        return out
