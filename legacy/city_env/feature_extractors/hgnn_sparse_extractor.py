from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from torch_geometric.nn import HeteroConv, GATConv, AttentionalAggregation, global_mean_pool, global_max_pool
from torch_geometric.data import HeteroData, Batch
import torch.nn.functional as F

class HGNNSparseExtractor(BaseFeaturesExtractor):
    def _requirements_matrix_to_key(self, requirements_matrix_batch):
        # Convert requirements_matrix_batch to a hashable key (bytes)
        # Assumes requirements_matrix_batch is a torch.Tensor on any device
        return requirements_matrix_batch.detach().cpu().numpy().tobytes()


    def __init__(self, 
                observation_space,
                features_dim=128, 
                pooling_types: list = None, # ['mean', 'max', 'attention', 'multi_head_attention']
                pooling_strategy='concat' # fusion or concat
                ):
        """
        Args:
            observation_space: gym Dict observation space
            features_dim: output feature dim
        """
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

        # Edge feature dimensions: [allocation_amount] for demand-resource edges and unallocated-resource edges
        self.edge_feat_dim_demand = 1  # for demand-resource, resource-demand
        self.edge_feat_dim_unalloc = 1  # for unallocated-resource, resource-unallocated

        # Make internal GNN sizes configurable so we can shrink the model easily.
        self.hidden_dim = 128
        self.gat_out = 32
        self.gat_heads = 4
        # pooling_types may be a string or list; normalize to list
        if pooling_types is None:
            pooling_types = ['attention']
        if isinstance(pooling_types, str):
            pooling_types = [pooling_types]
        self.pooling_types = pooling_types

        # Ensure at least one valid pooling type
        valid_poolings = {'attention', 'mean', 'max', 'multi_head_attention'}
        if not set(self.pooling_types).issubset(valid_poolings) or len(self.pooling_types) == 0:
            raise ValueError(f"pooling_types must be subset of {valid_poolings}; got {self.pooling_types}")

        # Ensure GAT output dimension times heads equals hidden_dim (as was 32*4=128 previously)
        if self.gat_out * self.gat_heads != self.hidden_dim:
            raise ValueError(f"gat_out * gat_heads must equal hidden_dim; got {self.gat_out}*{self.gat_heads}!={self.hidden_dim}")

        
        self.gat_norm = nn.ModuleDict({ # Not Applied, for Compatability
            ntype: nn.BatchNorm1d(self.hidden_dim) for ntype in ["demand", "resource", "unallocated"]
        })
        self.gat_dropout = nn.Dropout(p=0.2) # Not Applied, for Compatibility

        # Post-GNN projection layers for each node type (hidden_dim -> hidden_dim)
        self.proj_post_gnn = nn.ModuleDict({
            ntype: nn.Linear(self.hidden_dim, self.hidden_dim) for ntype in ["demand", "resource", "unallocated"]
        })

        self.hetero_conv1 = HeteroConv({
            ("demand", "demand_to_resource", "resource"): GATConv((self.demand_feat_dim, self.resource_feat_dim), self.gat_out, add_self_loops=False, heads=self.gat_heads, edge_dim=self.edge_feat_dim_demand),
            ("resource", "resource_to_demand", "demand"): GATConv((self.resource_feat_dim, self.demand_feat_dim), self.gat_out, add_self_loops=False, heads=self.gat_heads, edge_dim=self.edge_feat_dim_demand),
            ("unallocated", "unallocated_to_resource", "resource"): GATConv((self.unallocated_feat_dim, self.resource_feat_dim), self.gat_out, add_self_loops=False, heads=self.gat_heads, edge_dim=self.edge_feat_dim_unalloc),
            ("resource", "resource_to_unallocated", "unallocated"): GATConv((self.resource_feat_dim, self.unallocated_feat_dim), self.gat_out, add_self_loops=False, heads=self.gat_heads, edge_dim=self.edge_feat_dim_unalloc),
        }, aggr="sum")
        self.hetero_conv2 = HeteroConv({
            ("demand", "demand_to_resource", "resource"): GATConv((self.hidden_dim, self.hidden_dim), self.gat_out, add_self_loops=False, heads=self.gat_heads, edge_dim=self.edge_feat_dim_demand),
            ("resource", "resource_to_demand", "demand"): GATConv((self.hidden_dim, self.hidden_dim), self.gat_out, add_self_loops=False, heads=self.gat_heads, edge_dim=self.edge_feat_dim_demand),
            ("unallocated", "unallocated_to_resource", "resource"): GATConv((self.hidden_dim, self.hidden_dim), self.gat_out, add_self_loops=False, heads=self.gat_heads, edge_dim=self.edge_feat_dim_unalloc),
            ("resource", "resource_to_unallocated", "unallocated"): GATConv((self.hidden_dim, self.hidden_dim), self.gat_out, add_self_loops=False, heads=self.gat_heads, edge_dim=self.edge_feat_dim_unalloc),
        }, aggr="sum")

        # Preference-conditioned AttentionalAggregation pooling for each node type (only created if using 'attention')
        # Create attention modules only if attention pooling is requested
        if 'attention' in self.pooling_types:
            self.attention_pool = nn.ModuleDict({
                ntype: AttentionalAggregation(
                    gate_nn=nn.Sequential(
                        nn.Linear(self.hidden_dim + self.pref_dim, self.hidden_dim),
                        nn.SiLU(),
                        nn.Linear(self.hidden_dim, 1),
                    ),
                    nn=nn.Sequential(
                        nn.Linear(self.hidden_dim + self.pref_dim, self.hidden_dim),
                        nn.SiLU(),
                    ),
                ) for ntype in ["demand", "resource", "unallocated"]
            })
        else:
            self.attention_pool = None

        # Multi-Headed Attention Pooling for each node type (only created if using 'multi_head_attention')
        if 'multi_head_attention' in self.pooling_types:
            self.multi_head_attention_pool = nn.ModuleDict({
                ntype: nn.ModuleList([
                    AttentionalAggregation(
                        gate_nn=nn.Sequential(
                            nn.Linear(self.hidden_dim + self.pref_dim, self.hidden_dim),
                            nn.SiLU(),
                            nn.Linear(self.hidden_dim, 1),
                        ),
                        nn=nn.Sequential(
                            nn.Linear(self.hidden_dim + self.pref_dim, self.hidden_dim),
                            nn.SiLU(),
                        ),
                    ) for _ in range(self.gat_heads)
                ]) for ntype in ["demand", "resource", "unallocated"]
            })
        else:
            self.multi_head_attention_pool = None

        # LayerNorm per node type
        self.layer_norm = nn.ModuleDict({
            ntype: nn.LayerNorm(self.hidden_dim) for ntype in ["demand", "resource", "unallocated"]
        })

        # Build and cache static edge indices ONCE (assuming static graph structure)
        self._static_edge_indices = self._build_static_edge_indices()
        # Build and cache edge indices from requirements matrix ONCE (assuming static requirements)
        self._static_edge_indices_from_requirements = self._build_edge_indices_from_requirements(torch.ones((1, self.n_demands, self.n_resources)))

        # Cache for edge indices keyed by requirements matrix
        self._edge_indices_cache = {}

        # Add option for pooling fusion or concatenation (set early for final dim logic)
        self.pooling_strategy = pooling_strategy  # Options: 'fusion', 'concat'

        # Initialize pooling_gate as a ModuleDict so submodules move with .to(device)
        self.pooling_gate = nn.ModuleDict()

        # Modify pooling to use preference-conditioned gating mechanism
        if self.pooling_strategy == 'fusion':
            # Fusion strategy: produce a single hidden_dim vector per node type
            if 'multi_head_attention' in self.pooling_types:
                raise ValueError("multi_head_attention pooling is not currently supported with fusion strategy")
            for ntype in ["demand", "resource", "unallocated"]:
                self.pooling_gate[ntype] = nn.Sequential(
                    nn.Linear(self.pref_dim, len(self.pooling_types)),
                    nn.Softmax(dim=-1)
                )

        # ---------- Final projection layer (depends on pooling strategy) ----------
        if self.pooling_strategy == 'fusion':
            per_type_dim = self.hidden_dim
        elif self.pooling_strategy == 'concat':
            per_type_dim = 0
            for p in self.pooling_types:
                if p in ['attention', 'mean', 'max']:
                    per_type_dim += self.hidden_dim
                elif p == 'multi_head_attention':
                    per_type_dim += self.hidden_dim * self.gat_heads
                else:
                    raise ValueError(f"Unknown pooling type {p}")
        else:
            raise ValueError(f"Unknown pooling_strategy {self.pooling_strategy}")

        flat_input_dim = 3 * per_type_dim + self.pref_dim
        self.per_type_dim = per_type_dim
        self.final = nn.Sequential(
            nn.Linear(flat_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hidden_dim, features_dim),
        )

    def _compute_edge_features(self, obs, device):
        """Vectorized edge feature computation."""
        allocation_matrix_batch = obs["allocation_matrix"].to(device)
        requirements_matrix_batch = obs["requirements_matrix"].to(device)
        batch_size = allocation_matrix_batch.shape[0]
        n_demands, n_resources = self.n_demands, self.n_resources

        edge_attr_dict = {}

        # Demand to Resource edges (vectorized)
        if n_demands > 0:
            # Get all (batch, demand, resource) indices where requirements_matrix == 1
            batch_idx, d_idx, r_idx = torch.nonzero(requirements_matrix_batch, as_tuple=True)

            # Compute edge features: allocation amount for demand-resource edges
            edge_feats = allocation_matrix_batch[batch_idx, d_idx, r_idx].unsqueeze(1)  # Shape: (num_edges, 1)

            # Create edge attributes for demand_to_resource and resource_to_demand
            edge_attr_dict[("demand", "demand_to_resource", "resource")] = edge_feats
            edge_attr_dict[("resource", "resource_to_demand", "demand")] = edge_feats.clone()
        else:
            edge_attr_dict[("demand", "demand_to_resource", "resource")] = torch.empty(0, self.edge_feat_dim_demand, device=device)
            edge_attr_dict[("resource", "resource_to_demand", "demand")] = torch.empty(0, self.edge_feat_dim_demand, device=device)

        # Unallocated to Resource edges (vectorized)
        # Use only allocation amount as the edge feature for unallocated-resource edges
        unalloc_alloc = allocation_matrix_batch[:, n_demands, :].unsqueeze(-1)  # (B, R, 1)
        unalloc_edge_feats = unalloc_alloc.reshape(batch_size * n_resources, 1)
        edge_attr_dict[("unallocated", "unallocated_to_resource", "resource")] = unalloc_edge_feats
        edge_attr_dict[("resource", "resource_to_unallocated", "unallocated")] = unalloc_edge_feats.clone()

        return edge_attr_dict

    def _build_edge_indices_from_requirements(self, requirements_matrix_batch):
        # requirements_matrix_batch: (B, n_demands, n_resources)
        batch_size = requirements_matrix_batch.shape[0]
        n_demands, n_resources = self.n_demands, self.n_resources
        demand_to_resource = [[], []]
        resource_to_demand = [[], []]
        for b in range(batch_size):
            req_mat = requirements_matrix_batch[b]
            d_idx, r_idx = torch.where(req_mat == 1)
            # Offset for batch
            d_idx_b = d_idx + b * n_demands
            r_idx_b = r_idx + b * n_resources
            demand_to_resource[0].extend(d_idx_b.tolist())
            demand_to_resource[1].extend(r_idx_b.tolist())
            resource_to_demand[0].extend(r_idx_b.tolist())
            resource_to_demand[1].extend(d_idx_b.tolist())
        # Unallocated to Resource and reverse (fully connected per batch)
        unallocated_to_resource = [[], []]
        resource_to_unallocated = [[], []]
        for b in range(batch_size):
            for r in range(n_resources):
                unallocated_to_resource[0].append(b)  # unallocated node index (1 per batch)
                unallocated_to_resource[1].append(r + b * n_resources)
                resource_to_unallocated[0].append(r + b * n_resources)
                resource_to_unallocated[1].append(b)  # unallocated node index
        return {
            ("demand", "demand_to_resource", "resource"): torch.tensor(demand_to_resource, dtype=torch.long),
            ("resource", "resource_to_demand", "demand"): torch.tensor(resource_to_demand, dtype=torch.long),
            ("unallocated", "unallocated_to_resource", "resource"): torch.tensor(unallocated_to_resource, dtype=torch.long),
            ("resource", "resource_to_unallocated", "unallocated"): torch.tensor(resource_to_unallocated, dtype=torch.long),
        }

    def _build_static_edge_indices(self):
        """
        Build static edge indices based on the graph structure.
        """
        n_demands, n_resources = self.n_demands, self.n_resources
        demand_to_resource = [[], []]
        resource_to_demand = [[], []]
        for d in range(n_demands):
            for r in range(n_resources):
                demand_to_resource[0].append(d)
                demand_to_resource[1].append(r)
                resource_to_demand[0].append(r)
                resource_to_demand[1].append(d)

        unallocated_to_resource = ([0] * n_resources, [r for r in range(n_resources)])
        resource_to_unallocated = ([r for r in range(n_resources)], [0] * n_resources)

        return {
            ("demand", "demand_to_resource", "resource"): torch.tensor(demand_to_resource, dtype=torch.long),
            ("resource", "resource_to_demand", "demand"): torch.tensor(resource_to_demand, dtype=torch.long),
            ("unallocated", "unallocated_to_resource", "resource"): torch.tensor(unallocated_to_resource, dtype=torch.long),
            ("resource", "resource_to_unallocated", "unallocated"): torch.tensor(resource_to_unallocated, dtype=torch.long),
        }

    def forward(self, obs):
        device = next(self.parameters()).device
        allocation_matrix_batch = obs["allocation_matrix"].to(device)
        requirements_matrix_batch = obs["requirements_matrix"].to(device)
        prefs = obs["prefs"].to(device)
        # Ensure prefs tensor is on the same device as the model
        prefs = prefs.to(device)
        batch_size = allocation_matrix_batch.shape[0]
        n_demands, n_resources = self.n_demands, self.n_resources

        # Cache edge indices for each unique requirements matrix (per batch)
        req_key = self._requirements_matrix_to_key(requirements_matrix_batch)
        if req_key in self._edge_indices_cache:
            edge_indices = self._edge_indices_cache[req_key]
        else:
            edge_indices = self._build_edge_indices_from_requirements(requirements_matrix_batch)
            self._edge_indices_cache[req_key] = edge_indices

        # Compute edge features that encode requirements and allocations
        edge_attr_dict = self._compute_edge_features(obs, device)

        # Vectorized node feature construction for all batch elements
        # Demand node features
        if n_demands > 0:
            allocation_rows = allocation_matrix_batch[:, :n_demands, :]
            # Compute production per demand as the min allocation over required resources only.
            # Mask non-required resources so they don't affect the min.
            req = requirements_matrix_batch[:, :n_demands, :]  # (B, n_demands, n_resources)
            req_bool = req.to(dtype=torch.bool)
            # Use a large value for masked entries so they are ignored by min
            large_val = torch.finfo(allocation_rows.dtype).max / 2.0
            masked_alloc = torch.where(req_bool, allocation_rows, large_val * torch.ones_like(req_bool, dtype=allocation_rows.dtype))
            productions_raw = masked_alloc.min(dim=2)[0]  # Min across resource dimension
            has_req = req_bool.any(dim=2)  # (B, n_demands); True where at least one resource is required
            productions = torch.where(has_req, productions_raw, torch.zeros_like(productions_raw))
            productions = productions.unsqueeze(2)  # (B, n_demands, 1)
            prefs_expanded = prefs.unsqueeze(1).expand(-1, n_demands, -1)    # (B, n_demands, pref_dim)
            demand_feats = torch.cat([productions, allocation_rows, prefs_expanded], dim=2)  # (B, n_demands, d_feat)
            demand_feats = demand_feats.reshape(-1, self.demand_feat_dim)  # (B*n_demands, d_feat)
        else:
            demand_feats = torch.empty(0, self.demand_feat_dim, device=device)
        # Resource node features (fix: available is a single value per resource, not a vector)
        available = allocation_matrix_batch[:, n_demands, :].unsqueeze(2)  # (B, n_resources, 1)
        allocation_cols = allocation_matrix_batch[:, :n_demands, :].permute(0, 2, 1)  # (B, n_resources, n_demands)
        prefs_expanded = prefs.unsqueeze(1).expand(-1, n_resources, -1)  # (B, n_resources, pref_dim)
        resource_feats = torch.cat([available, allocation_cols, prefs_expanded], dim=2)  # (B, n_resources, r_feat)
        resource_feats = resource_feats.reshape(-1, self.resource_feat_dim)  # (B*n_resources, r_feat)
        # Unallocated node features
        unallocated = allocation_matrix_batch[:, n_demands, :].unsqueeze(1)  # (B, 1, n_resources)
        zero_prod = torch.zeros(batch_size, 1, 1, device=device)
        prefs_single = prefs.unsqueeze(1)  # (B, 1, pref_dim)
        unallocated_feats = torch.cat([zero_prod, unallocated, prefs_single], dim=2)  # (B, 1, u_feat)
        unallocated_feats = unallocated_feats.reshape(-1, self.unallocated_feat_dim)  # (B, 1, u_feat)

        # Build a single HeteroData batch with static edge indices, repeated for each batch
        data = HeteroData()
        data['demand'].x = demand_feats
        data['resource'].x = resource_feats
        data['unallocated'].x = unallocated_feats
        # Set up batch vector for each node type
        for ntype in ["demand", "resource", "unallocated"]:
            n_nodes = {"demand": n_demands, "resource": n_resources, "unallocated": 1}[ntype]
            if n_nodes > 0:
                data[ntype].batch = torch.arange(batch_size, device=device).repeat_interleave(n_nodes)
            else:
                data[ntype].batch = torch.empty(0, dtype=torch.long, device=device)
        # Set edge indices and edge features
        for k, edge_idx in edge_indices.items():
            data[k].edge_index = edge_idx.to(device)
            data[k].edge_attr = edge_attr_dict[k]

        x_dict = data.x_dict
        edge_index_dict = {k: data[k].edge_index for k in edge_indices.keys()}
        edge_attr_dict = {k: data[k].edge_attr for k in edge_indices.keys()}
        # First HeteroConv layer (raw features)
        x1 = self.hetero_conv1(x_dict, edge_index_dict, edge_attr_dict)
        # Second HeteroConv layer (hidden_dim-dim)
        x2 = self.hetero_conv2(x1, edge_index_dict, edge_attr_dict)
        # Residual connection: add output of first layer to second, followed by activation
        x_out = {}
        for ntype in ["demand", "resource", "unallocated"]:
            if x1[ntype].size(0) > 0 and x2[ntype].size(0) > 0:
                x_out[ntype] = F.silu(x2[ntype] + x1[ntype])  # Single residual path with activation
            elif x2[ntype].size(0) > 0:
                x_out[ntype] = F.silu(x2[ntype])
            else:
                x_out[ntype] = torch.zeros(0, self.hidden_dim, device=device)

        # Apply pre-created LayerNorms
        for ntype in x_out:
            if x_out[ntype].size(0) > 0:
                x_out[ntype] = self.layer_norm[ntype](x_out[ntype])

        # Pooling per node type: support multiple of 'attention', 'mean', 'max'
        pooled_per_type = []
        for ntype in ["demand", "resource", "unallocated"]:
            x_nodes = x_out[ntype]
            if x_nodes.size(0) > 0:
                batch_idx = data[ntype].batch if hasattr(data[ntype], 'batch') else torch.zeros(x_nodes.size(0), dtype=torch.long, device=device)
                pooled_for_ntype = []
                for p in self.pooling_types:
                    if p == 'attention':
                        prefs_per_node = prefs[batch_idx]
                        x_with_prefs = torch.cat([x_nodes, prefs_per_node], dim=1)
                        pooled_nt = self.attention_pool[ntype](x_with_prefs, batch_idx)
                    elif p == 'multi_head_attention':
                        prefs_per_node = prefs[batch_idx]
                        x_with_prefs = torch.cat([x_nodes, prefs_per_node], dim=1)
                        pooled_heads = [head(x_with_prefs, batch_idx) for head in self.multi_head_attention_pool[ntype]]
                        pooled_nt = torch.cat(pooled_heads, dim=1)  # Concatenate results from all heads
                    elif p == 'mean':
                        pooled_nt = global_mean_pool(x_nodes, batch_idx)
                    elif p == 'max':
                        pooled_nt = global_max_pool(x_nodes, batch_idx)
                    else:
                        raise ValueError(f"Unknown pooling: {p}")
                    pooled_for_ntype.append(pooled_nt)
                if self.pooling_strategy == 'fusion':
                    # Correct fusion logic for pooling
                    pooled_stack = torch.stack(pooled_for_ntype, dim=1)  # (B, num_poolings, hidden_dim)
                    gate_weights = self.pooling_gate[ntype](prefs)       # (B, num_poolings)
                    gate_weights = gate_weights.unsqueeze(-1)            # (B, num_poolings, 1)

                    # Weighted sum across poolings
                    pooled_nt = (pooled_stack * gate_weights).sum(dim=1)  # (B, hidden_dim)
                elif self.pooling_strategy == 'concat':
                    # Concatenate pooling results for this node type along feature dim
                    pooled_nt = torch.cat(pooled_for_ntype, dim=1)
                pooled_per_type.append(pooled_nt)
            else:
                # If no nodes of this type, append zeros matching expected per-type dim
                pooled_per_type.append(torch.zeros(batch_size, self.per_type_dim, device=device))
        # Concatenate pooled results across node types (demand, resource, unallocated)
        pooled_all = torch.cat(pooled_per_type, dim=1)
        x = torch.cat([pooled_all, prefs], dim=-1)
        out = self.final(x)
        return out