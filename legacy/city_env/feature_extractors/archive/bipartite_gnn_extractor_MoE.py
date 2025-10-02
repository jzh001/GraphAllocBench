from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from torch_geometric.nn import HeteroConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import HeteroData, Batch
import torch.nn.functional as F

class BipartiteGNNExtractorMoE(BaseFeaturesExtractor): # based on V12
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

        # FiLM modulation layers for each node type (after first GNN layer)
        self.film_layers = nn.ModuleDict({
            ntype: nn.Linear(self.pref_dim, 128)  # 64 gamma, 64 beta
            for ntype in ["demand", "resource", "unallocated"]
        })

        # Projection layers for skip connections (input features to 64-dim)
        self.proj = nn.ModuleDict({
            "demand": nn.Linear(self.demand_feat_dim, 64),
            "resource": nn.Linear(self.resource_feat_dim, 64),
            "unallocated": nn.Linear(self.unallocated_feat_dim, 64),
        })

        # Two-layer GATConv for each edge type (no self loops)
        self.hetero_conv1 = HeteroConv({
            ("demand", "demand_to_resource", "resource"): GATConv((self.demand_feat_dim, self.resource_feat_dim), 64, add_self_loops=False),
            ("resource", "resource_to_demand", "demand"): GATConv((self.resource_feat_dim, self.demand_feat_dim), 64, add_self_loops=False),
            ("unallocated", "unallocated_to_resource", "resource"): GATConv((self.unallocated_feat_dim, self.resource_feat_dim), 64, add_self_loops=False),
            ("resource", "resource_to_unallocated", "unallocated"): GATConv((self.resource_feat_dim, self.unallocated_feat_dim), 64, add_self_loops=False),
        }, aggr="sum")
        self.hetero_conv2 = HeteroConv({
            ("demand", "demand_to_resource", "resource"): GATConv((64, 64), 64, add_self_loops=False),
            ("resource", "resource_to_demand", "demand"): GATConv((64, 64), 64, add_self_loops=False),
            ("unallocated", "unallocated_to_resource", "resource"): GATConv((64, 64), 64, add_self_loops=False),
            ("resource", "resource_to_unallocated", "unallocated"): GATConv((64, 64), 64, add_self_loops=False),
        }, aggr="sum")

        # LayerNorm for each node type after GNN
        self.norms = nn.ModuleDict({
            "demand": nn.LayerNorm(64),
            "resource": nn.LayerNorm(64),
            "unallocated": nn.LayerNorm(64),
        })

        self.final = nn.Sequential(
            nn.LayerNorm(128 * 3 + self.pref_dim),
            nn.ReLU(),
            nn.Linear(128 * 3 + self.pref_dim, features_dim)
        )

        # MoE for each node type (after GNN, before pooling)
        self.moe = nn.ModuleDict({
            ntype: SparseMoE(64, self.pref_dim, n_experts=4, hidden_dim=64)
            for ntype in ["demand", "resource", "unallocated"]
        })

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
        # First GNN layer
        x_out1 = self.hetero_conv1(x_dict, batch.edge_index_dict)
        for ntype in ["demand", "resource", "unallocated"]:
            if x_out1[ntype].size(0) > 0:
                x_out1[ntype] = F.relu(x_out1[ntype])
            else:
                x_out1[ntype] = torch.zeros(0, 64, device=device)

        # FiLM modulation: modulate x_out1 for each node type using prefs
        film_x_out1 = {}
        for ntype in ["demand", "resource", "unallocated"]:
            x = x_out1[ntype]
            if x.size(0) > 0:
                # For each node, get the corresponding prefs vector (batch assignment)
                batch_idx = batch[ntype].batch  # shape: (num_nodes,)
                # Gather prefs for each node in this type
                node_prefs = prefs[batch_idx]  # shape: (num_nodes, pref_dim)
                film_params = self.film_layers[ntype](node_prefs)  # (num_nodes, 128)
                gamma, beta = film_params[:, :64], film_params[:, 64:]
                film_x_out1[ntype] = gamma * x + beta
            else:
                film_x_out1[ntype] = x

        # Save skip connection input for each node type
        skip_inputs = {}
        for ntype in ["demand", "resource", "unallocated"]:
            if x_dict[ntype].size(0) > 0:
                skip_inputs[ntype] = self.proj[ntype](x_dict[ntype])
            else:
                skip_inputs[ntype] = torch.zeros(0, 64, device=device)

        # Second GNN layer (input is FiLM-modulated x_out1)
        x_out2 = self.hetero_conv2(film_x_out1, batch.edge_index_dict)
        # Add skip connection and apply LayerNorm after second layer
        for ntype in ["demand", "resource", "unallocated"]:
            if x_out2[ntype].size(0) > 0:
                x_out2[ntype] = self.norms[ntype](F.relu(x_out2[ntype] + skip_inputs[ntype]))
            else:
                x_out2[ntype] = torch.zeros(0, 64, device=device)
        # --- MoE enhancement: apply per-node-type MoE, conditioned on prefs ---
        pooled = []
        for ntype in ["demand", "resource", "unallocated"]:
            x_nodes = x_out2[ntype]
            if x_nodes.size(0) > 0:
                # Gather prefs for each node in this type (batch assignment)
                batch_idx = batch[ntype].batch  # (num_nodes,)
                node_prefs = prefs[batch_idx]  # (num_nodes, pref_dim)
                moe_out = self.moe[ntype](x_nodes, node_prefs)
                mean_pooled = global_mean_pool(moe_out, batch[ntype].batch)
                max_pooled = global_max_pool(moe_out, batch[ntype].batch)
            else:
                mean_pooled = torch.zeros(batch_size, 64, device=device)
                max_pooled = torch.zeros(batch_size, 64, device=device)
            pooled.append(torch.cat([mean_pooled, max_pooled], dim=1))
        pooled_all = torch.cat(pooled, dim=1)
        x = torch.cat([pooled_all, prefs], dim=-1)
        out = self.final(x)
        return out

# --- Mixture of Experts (MoE) block ---
class SparseMoE(nn.Module):
    def __init__(self, input_dim, pref_dim, n_experts=4, hidden_dim=64):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(n_experts)
        ])
        # Gating network: input is [node_feat, prefs]
        self.gate = nn.Sequential(
            nn.Linear(input_dim + pref_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts)
        )

    def forward(self, x, prefs):
        # x: (num_nodes, input_dim), prefs: (num_nodes, pref_dim)
        gate_input = torch.cat([x, prefs], dim=-1)
        logits = self.gate(gate_input)
        # Sparse softmax: top-1 expert (can use top-2 for more capacity)
        top1 = torch.argmax(logits, dim=-1)  # (num_nodes,)
        out = torch.zeros(x.size(0), self.experts[0][-1].out_features, device=x.device)
        for i in range(self.n_experts):
            mask = (top1 == i)
            if mask.any():
                out[mask] = self.experts[i](x[mask])
        return out