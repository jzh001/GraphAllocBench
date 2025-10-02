from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from torch_geometric.nn import HeteroConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import HeteroData, Batch
import torch.nn.functional as F

class BipartiteGNNExtractorMoEV6(BaseFeaturesExtractor): # based on V12
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

        # --- Preference embedding ---
        self.pref_embedding_dim = 32  # Increased embedding dim for more capacity
        self.pref_embedding = nn.Sequential(
            nn.Linear(self.pref_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.pref_embedding_dim),
            nn.ReLU()
        )

        # Deeper FiLM MLPs for each node type (after first GNN layer)
        def make_film_mlp(emb_dim):
            return nn.Sequential(
                nn.Linear(emb_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 128)  # 64 gamma, 64 beta
            )
        self.film_layers = nn.ModuleDict({
            ntype: make_film_mlp(self.pref_embedding_dim)
            for ntype in ["demand", "resource", "unallocated"]
        })
        # Final FiLM layer (after MoE, before pooling)
        self.final_film_layers = nn.ModuleDict({
            ntype: make_film_mlp(self.pref_embedding_dim)
            for ntype in ["demand", "resource", "unallocated"]
        })
        # LayerNorm after FiLM for each node type
        self.film_norms = nn.ModuleDict({
            ntype: nn.LayerNorm(64) for ntype in ["demand", "resource", "unallocated"]
        })
        self.final_film_norms = nn.ModuleDict({
            ntype: nn.LayerNorm(64) for ntype in ["demand", "resource", "unallocated"]
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
            nn.Linear(128 * 3 + self.pref_dim, max(features_dim, 256)),
            nn.ReLU(),
            nn.Linear(max(features_dim, 256), features_dim)
        )

        # MoE for each node type (after GNN, before pooling)
        self.moe = nn.ModuleDict({
            ntype: SparseMoE(64, self.pref_embedding_dim, n_experts=8, hidden_dim=64)
            for ntype in ["demand", "resource", "unallocated"]
        })

        # Preference-query attention pooling for each node type (with non-linearity)
        # Input is [node_feat, prefs_emb_query] (64 + pref_embedding_dim)
        self.attn_pool = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(64 + self.pref_embedding_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            for ntype in ["demand", "resource", "unallocated"]
        })

        # --- Caching for graph and prefs ---
        self._cached_graph = None
        self._cached_prefs = None
        self._cached_alloc = None
        self._cached_reqs = None
        self._cached_batch_size = None

    def _build_graph_batch(self, allocation_matrix_batch, requirements_matrix_batch, prefs, device):
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
        return Batch.from_data_list(data_list)

    def forward(self, obs):
        device = next(self.parameters()).device
        allocation_matrix_batch = obs["allocation_matrix"].to(device)
        requirements_matrix_batch = obs["requirements_matrix"].to(device)
        prefs = obs["prefs"].to(device)

        batch_size = allocation_matrix_batch.shape[0]

        # --- Caching logic: only rebuild graph if alloc/reqs/prefs change ---
        cache_hit = (
            self._cached_graph is not None and
            self._cached_batch_size == batch_size and
            torch.equal(allocation_matrix_batch, self._cached_alloc) and
            torch.equal(requirements_matrix_batch, self._cached_reqs) and
            torch.equal(prefs, self._cached_prefs)
        )
        if cache_hit:
            batch = self._cached_graph
        else:
            batch = self._build_graph_batch(allocation_matrix_batch, requirements_matrix_batch, prefs, device)
            self._cached_graph = batch
            self._cached_alloc = allocation_matrix_batch.clone()
            self._cached_reqs = requirements_matrix_batch.clone()
            self._cached_prefs = prefs.clone()
            self._cached_batch_size = batch_size

        x_dict = batch.x_dict
        # First GNN layer
        x_out1 = self.hetero_conv1(x_dict, batch.edge_index_dict)
        for ntype in ["demand", "resource", "unallocated"]:
            if x_out1[ntype].size(0) > 0:
                x_out1[ntype] = F.relu(x_out1[ntype])
            else:
                x_out1[ntype] = torch.zeros(0, 64, device=device)

        # FiLM modulation: modulate x_out1 for each node type using embedded prefs
        film_x_out1 = {}
        for ntype in ["demand", "resource", "unallocated"]:
            x = x_out1[ntype]
            if x.size(0) > 0:
                batch_idx = batch[ntype].batch  # shape: (num_nodes,)
                node_prefs = prefs[batch_idx]  # shape: (num_nodes, pref_dim)
                node_prefs_emb = self.pref_embedding(node_prefs)
                film_params = self.film_layers[ntype](node_prefs_emb)  # (num_nodes, 128)
                gamma, beta = film_params[:, :64], film_params[:, 64:]
                # Residual FiLM: out = x + gamma * x + beta
                x_film = x + gamma * x + beta
                x_film = self.film_norms[ntype](x_film)
                film_x_out1[ntype] = x_film
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
        # --- MoE enhancement: apply per-node-type MoE, conditioned on embedded prefs ---
        pooled = []
        for ntype in ["demand", "resource", "unallocated"]:
            x_nodes = x_out2[ntype]
            if x_nodes.size(0) > 0:
                batch_idx = batch[ntype].batch  # (num_nodes,)
                node_prefs = prefs[batch_idx]  # (num_nodes, pref_dim)
                node_prefs_emb = self.pref_embedding(node_prefs)
                moe_out = self.moe[ntype](x_nodes, node_prefs_emb)
                film_params = self.final_film_layers[ntype](node_prefs_emb)  # (num_nodes, 128)
                gamma, beta = film_params[:, :64], film_params[:, 64:]
                # Residual FiLM: out = x + gamma * x + beta
                film_moe_out = moe_out + gamma * moe_out + beta
                film_moe_out = self.final_film_norms[ntype](film_moe_out)

                # --- Vectorized attention pooling ---
                num_graphs = batch_size
                N = film_moe_out.size(0) // num_graphs
                feat_dim = film_moe_out.size(1)
                film_moe_out_reshaped = film_moe_out.view(num_graphs, N, feat_dim)
                # Use per-graph preference embedding as query for all nodes in the graph
                prefs_emb_query = self.pref_embedding(prefs).unsqueeze(1).expand(num_graphs, N, self.pref_embedding_dim)  # (num_graphs, N, emb_dim)
                attn_input = torch.cat([film_moe_out_reshaped, prefs_emb_query], dim=2)  # (num_graphs, N, feat_dim+emb_dim)
                # Update attn_pool input dim if needed
                attn_logits = self.attn_pool[ntype](attn_input).squeeze(-1)  # (num_graphs, N)
                attn_weights = torch.softmax(attn_logits, dim=1)  # (num_graphs, N)
                attn_pooled = (film_moe_out_reshaped * attn_weights.unsqueeze(-1)).sum(dim=1)  # (num_graphs, feat_dim)
                max_pooled = film_moe_out_reshaped.max(dim=1).values  # (num_graphs, feat_dim)
            else:
                attn_pooled = torch.zeros(batch_size, 64, device=device)
                max_pooled = torch.zeros(batch_size, 64, device=device)
            pooled.append(torch.cat([attn_pooled, max_pooled], dim=1))
        pooled_all = torch.cat(pooled, dim=1)
        x = torch.cat([pooled_all, prefs], dim=-1)
        out = self.final(x)
        return out

# --- Mixture of Experts (MoE) block ---
class SparseMoE(nn.Module):
    def __init__(self, input_dim, pref_dim, n_experts=4, hidden_dim=64):
        super().__init__()
        self.n_experts = n_experts
        # Each expert is now a two-layer MLP, with FiLM after the first layer
        self.expert_fc1 = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(n_experts)
        ])
        self.expert_fc2 = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_experts)
        ])
        # FiLM MLP for each expert (can be shared, but here per-expert)
        def make_film_linear(pref_dim):
            return nn.Linear(pref_dim, 128)  # 64 gamma, 64 beta
        self.film_mlps = nn.ModuleList([
            make_film_linear(pref_dim) for _ in range(n_experts)
        ])
        # Gating network: input is [node_feat, prefs] (now a small MLP)
        self.gate = nn.Sequential(
            nn.Linear(input_dim + pref_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts)
        )

    def forward(self, x, prefs, noise_std=1.0, top_k=2):
        # x: (num_nodes, input_dim), prefs: (num_nodes, pref_dim)
        gate_input = torch.cat([x, prefs], dim=-1)
        logits = self.gate(gate_input)
        # Add Gaussian noise to logits for noisy top-k routing
        if self.training and noise_std > 0:
            noise = torch.randn_like(logits) * noise_std
            noisy_logits = logits + noise
        else:
            noisy_logits = logits
        # Select top-k experts for each node
        topk = torch.topk(noisy_logits, k=top_k, dim=-1)
        topk_indices = topk.indices  # (num_nodes, top_k)
        out = torch.zeros(x.size(0), self.expert_fc2[0].out_features, device=x.device)
        # For each expert, process nodes assigned to it (could be in top-k for multiple nodes)
        for i in range(self.n_experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                x_i = x[mask]
                prefs_i = prefs[mask]
                # First layer
                h = self.expert_fc1[i](x_i)
                # FiLM modulation: gamma, beta from prefs
                film_params = self.film_mlps[i](prefs_i)  # (num_nodes, 128)
                gamma, beta = film_params[:, :64], film_params[:, 64:]
                h = gamma * h + beta
                h = F.relu(h)
                # Second layer
                expert_out = self.expert_fc2[i](h)
                # For nodes with multiple experts in top-k, split output equally (simple average)
                count = (topk_indices[mask] == i).sum(dim=-1, keepdim=True).float()
                expert_out = expert_out / count
                out[mask] += expert_out
        return out