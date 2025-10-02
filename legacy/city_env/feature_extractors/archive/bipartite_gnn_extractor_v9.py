from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from torch_geometric.nn import HeteroConv, GCNConv, GATConv, GlobalAttention, SAGEConv
from torch_geometric.data import HeteroData, Batch
import torch.nn.functional as F

# Define the FiLMModulator
class FiLMModulator(nn.Module):
    def __init__(self, pref_dim, embed_dim, hidden_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        # Network to predict gamma and beta from preferences
        self.net = nn.Sequential(
            nn.Linear(pref_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * embed_dim) # Output 2 * embed_dim for gamma and beta
        )

    def forward(self, prefs): # prefs: (batch_size, pref_dim)
        gamma_beta = self.net(prefs) # (batch_size, 2 * embed_dim)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1) # (batch_size, embed_dim) for each
        return gamma, beta

class TransformerAttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads, pref_dim):
        super(TransformerAttentionPooling, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.query_generator = nn.Sequential(
            nn.Linear(pref_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.activation = nn.SiLU()

    def forward(self, x, batch, prefs=None):
        if prefs is None:
            raise ValueError("Preferences must be provided to preference-conditioned TransformerAttentionPooling")

        batch_size = batch.max().item() + 1
        num_nodes_per_graph = torch.bincount(batch)

        # Pad sequences for MultiheadAttention
        # Using a fixed max_nodes for simplicity, better to use dynamic padding/masking for efficiency
        max_nodes = num_nodes_per_graph.max().item() # Get max nodes in current batch
        reshaped_x = torch.zeros(batch_size, max_nodes, x.size(-1), device=x.device)
        key_padding_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=x.device) # True for padding
        
        for i in range(batch_size):
            graph_nodes = x[batch == i]
            reshaped_x[i, :graph_nodes.size(0)] = graph_nodes
            key_padding_mask[i, :graph_nodes.size(0)] = False # False for actual data

        query = self.query_generator(prefs).unsqueeze(1) 

        # Pass key_padding_mask to multihead_attn
        attn_output, _ = self.multihead_attn(query, reshaped_x, reshaped_x, key_padding_mask=key_padding_mask)
        attn_output = self.activation(attn_output)

        return attn_output.squeeze(1)

class BipartiteGNNExtractorV9(BaseFeaturesExtractor):
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

        self.embed_dim = 64 # Define embed_dim for clarity

        # Feature dims for each node type (no padding)
        self.demand_feat_dim = 1 + self.n_resources + self.pref_dim  # production + allocation_row + prefs
        self.resource_feat_dim = 1 + self.n_demands + self.pref_dim  # available + allocation_col + prefs
        self.unallocated_feat_dim = 1 + self.n_resources + self.pref_dim  # 0.0 + unallocated + prefs

        # Stack two HeteroConv layers for deeper message passing
        self.hetero_convs = nn.ModuleList([
            HeteroConv({
                ("demand", "demand_to_resource", "resource"): SAGEConv((self.demand_feat_dim, self.resource_feat_dim), self.embed_dim),
                ("resource", "resource_to_demand", "demand"): SAGEConv((self.resource_feat_dim, self.demand_feat_dim), self.embed_dim),
                ("unallocated", "unallocated_to_resource", "resource"): SAGEConv((self.unallocated_feat_dim, self.resource_feat_dim), self.embed_dim),
                ("resource", "resource_to_unallocated", "unallocated"): SAGEConv((self.resource_feat_dim, self.unallocated_feat_dim), self.embed_dim),
            }, aggr="sum"),
            HeteroConv({
                ("demand", "demand_to_resource", "resource"): GATConv((self.embed_dim, self.embed_dim), self.embed_dim, heads=2, concat=False, add_self_loops=False),
                ("resource", "resource_to_demand", "demand"): GATConv((self.embed_dim, self.embed_dim), self.embed_dim, heads=2, concat=False, add_self_loops=False),
                ("unallocated", "unallocated_to_resource", "resource"): GATConv((self.embed_dim, self.embed_dim), self.embed_dim, heads=2, concat=False, add_self_loops=False),
                ("resource", "resource_to_unallocated", "unallocated"): GATConv((self.embed_dim, self.embed_dim), self.embed_dim, heads=2, concat=False, add_self_loops=False),
            }, aggr="sum"),
        ])

        # Dropout after each HeteroConv layer
        self.dropout = nn.Dropout(0.1)

        # Layer Normalization for each node type
        self.norm = nn.ModuleDict({
            "demand": nn.LayerNorm(self.embed_dim),
            "resource": nn.LayerNorm(self.embed_dim),
            "unallocated": nn.LayerNorm(self.embed_dim),
        })

        # FiLM Modulators for each node type and each GNN layer (if desired)
        # Here we'll apply it after LayerNorm and SiLU in the loop
        self.film_modulators = nn.ModuleDict({
            "demand": FiLMModulator(self.pref_dim, self.embed_dim),
            "resource": FiLMModulator(self.pref_dim, self.embed_dim),
            "unallocated": FiLMModulator(self.pref_dim, self.embed_dim),
        })

        # Attention pooling for each node type
        self.att_pool = TransformerAttentionPooling(embed_dim=self.embed_dim, num_heads=4, pref_dim=self.pref_dim)

        # Final MLP
        self.linear_graph = nn.Linear(self.embed_dim * 3 + self.pref_dim, 128)  # 3 node types pooled + prefs
        self.final = nn.Linear(128, features_dim)


    def forward(self, obs):
        device = next(self.parameters()).device
        allocation_matrix_batch = obs["allocation_matrix"].to(device)
        requirements_matrix_batch = obs["requirements_matrix"].to(device)
        prefs = obs["prefs"].to(device)

        batch_size, n_demands_plus1, n_resources = allocation_matrix_batch.shape
        n_demands = self.n_demands
        n_resources = self.n_resources

        # Build HeteroData objects for the batch (same as before)
        data_list = []
        for i in range(batch_size):
            # Node features
            if n_demands > 0:
                allocation_rows = allocation_matrix_batch[i, :n_demands, :]
                productions = torch.min(allocation_rows, dim=1)[0].unsqueeze(1)
                prefs_expanded = prefs[i].unsqueeze(0).expand(n_demands, -1)
                demand_feats = torch.cat([productions, allocation_rows, prefs_expanded], dim=1)
            else:
                demand_feats = torch.empty(0, self.demand_feat_dim, device=device, dtype=torch.float)

            available = allocation_matrix_batch[i, n_demands, :].unsqueeze(1)
            allocation_cols = allocation_matrix_batch[i, :n_demands, :].T
            prefs_expanded = prefs[i].unsqueeze(0).expand(n_resources, -1)
            resource_feats = torch.cat([available, allocation_cols, prefs_expanded], dim=1)

            unallocated = allocation_matrix_batch[i, n_demands, :].unsqueeze(0)
            zero_prod = torch.zeros(1, 1, device=device, dtype=torch.float)
            prefs_single = prefs[i].unsqueeze(0)
            unallocated_feats = torch.cat([zero_prod, unallocated, prefs_single], dim=1)

            # Edge indices and weights (same as before)
            requirements_matrix = requirements_matrix_batch[i].tolist()
            demand_to_resource = [[], []]
            resource_to_demand = [[], []]
            demand_to_resource_weights = []
            resource_to_demand_weights = []
            for d in range(n_demands):
                for r in range(n_resources):
                    if requirements_matrix[d][r] == 1:
                        demand_to_resource[0].append(d)
                        demand_to_resource[1].append(r)
                        resource_to_demand[0].append(r)
                        resource_to_demand[1].append(d)
                        allocation_weight = allocation_matrix_batch[i, d, r].item()
                        demand_to_resource_weights.append(allocation_weight)
                        resource_to_demand_weights.append(allocation_weight)

            unallocated_to_resource = [[], []]
            resource_to_unallocated = [[], []]
            unallocated_to_resource_weights = []
            resource_to_unallocated_weights = []
            for r in range(n_resources):
                unallocated_to_resource[0].append(0)
                unallocated_to_resource[1].append(r)
                resource_to_unallocated[0].append(r)
                resource_to_unallocated[1].append(0)
                unallocated_weight = allocation_matrix_batch[i, n_demands, r].item()
                unallocated_to_resource_weights.append(unallocated_weight)
                resource_to_unallocated_weights.append(unallocated_weight)

            # Build HeteroData object
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
            
            # Add edge weights
            if demand_to_resource_weights:
                data['demand', 'demand_to_resource', 'resource'].edge_weight = torch.tensor(demand_to_resource_weights, dtype=torch.float, device=device)
            if resource_to_demand_weights:
                data['resource', 'resource_to_demand', 'demand'].edge_weight = torch.tensor(resource_to_demand_weights, dtype=torch.float, device=device)
            if unallocated_to_resource_weights:
                data['unallocated', 'unallocated_to_resource', 'resource'].edge_weight = torch.tensor(unallocated_to_resource_weights, dtype=torch.float, device=device)
            if resource_to_unallocated_weights:
                data['resource', 'resource_to_unallocated', 'unallocated'].edge_weight = torch.tensor(resource_to_unallocated_weights, dtype=torch.float, device=device)
            
            data_list.append(data)

        # Batch HeteroData objects
        batch = Batch.from_data_list(data_list)

        # --- Forward pass through stacked HeteroConv layers ---
        x_dict = batch.x_dict
        # SAGEConv and GATConv as used here do not directly take edge_weight,
        # so edge_weight_dict is extracted but not used in the convs themselves.
        # As discussed, if edge_weight is critical for message strength,
        # you might need to use a different conv layer (e.g., GCNConv) or
        # inject aggregated edge weight info into node features.
        edge_weight_dict = batch.edge_weight_dict

        # First HeteroConv layer
        x1 = self.hetero_convs[0](x_dict, batch.edge_index_dict)
        for ntype in x1:
            # Apply residual connection if shapes match (x_dict[ntype] is the initial feature here)
            if x1[ntype].shape == x_dict[ntype].shape:
                x1[ntype] = x1[ntype] + x_dict[ntype]
            
            # Apply LayerNorm
            normalized_x = self.norm[ntype](x1[ntype])
            
            # Apply FiLM
            gamma_batch, beta_batch = self.film_modulators[ntype](prefs)
            gamma_expanded = gamma_batch[batch[ntype].batch]
            beta_expanded = beta_batch[batch[ntype].batch]
            x1[ntype] = gamma_expanded * normalized_x + beta_expanded
            
            x1[ntype] = F.silu(x1[ntype])
            x1[ntype] = self.dropout(x1[ntype])

        # Second HeteroConv layer (GATConv), with residual connection
        x2 = self.hetero_convs[1](x1, batch.edge_index_dict)
        for ntype in x2:
            # Apply residual connection
            if x2[ntype].shape == x1[ntype].shape:
                x2[ntype] = x2[ntype] + x1[ntype]
            
            # Apply LayerNorm
            normalized_x = self.norm[ntype](x2[ntype])
            
            # Apply FiLM
            gamma_batch, beta_batch = self.film_modulators[ntype](prefs)
            gamma_expanded = gamma_batch[batch[ntype].batch]
            beta_expanded = beta_batch[batch[ntype].batch]
            x2[ntype] = gamma_expanded * normalized_x + beta_expanded
            
            x2[ntype] = F.silu(x2[ntype])
            x2[ntype] = self.dropout(x2[ntype])
        x_dict = x2 # Update x_dict to be the output of the final GNN layer

        # --- Attention pooling for each node type using batch vector ---
        pooled = []
        for ntype in ["demand", "resource", "unallocated"]:
            if x_dict[ntype].size(0) > 0:
                # Pass prefs to the pooling layer
                pooled_ntype = self.att_pool(x_dict[ntype], batch=batch[ntype].batch, prefs=prefs)
            else:
                pooled_ntype = torch.zeros(batch_size, self.att_pool.query_generator[-1].out_features, device=device, dtype=torch.float)
            pooled.append(pooled_ntype)
        # Concatenate all pooled node type embeddings for each batch item
        pooled_all = torch.cat(pooled, dim=1)  # (batch_size, embed_dim * 3)

        # Concatenate with preferences: both should be 2D tensors
        x = torch.cat([pooled_all, prefs], dim=-1)
        x = self.linear_graph(x)
        out = self.final(x)
        return out