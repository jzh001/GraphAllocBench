import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch
from torch import nn
import torch.nn.functional as F

class PrefAttnBias(nn.Module):
    """
    Tiny MLP that turns (prefs, edge_attr) into a scalar bias per edge for attention logits.
    Near-zero initialized so training starts from the base GAT behavior.
    """
    def __init__(self, pref_dim: int, edge_dim: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pref_dim + edge_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, prefs_per_edge: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # prefs_per_edge: (E, P), edge_attr: (E, edge_dim)
        x = torch.cat([prefs_per_edge, edge_attr], dim=-1)
        return self.mlp(x).squeeze(-1)  # (E,)
    

class LoRAAdapter(nn.Module):
    """
    Low-rank additive update for a weight matrix W: W' = W + A diag(s(p)) B.
    We generate s(p) from prefs via a tiny MLP. Near-zero initialized.
    """
    def __init__(self, d_out: int, d_in: int, r: int, pref_dim: int, mlp_hidden: int = 32):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d_out, r) * 0.01)
        self.B = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.gen = nn.Sequential(
            nn.Linear(pref_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, r)
        )
        # zero-init generator’s last layer so initial update ~ 0
        nn.init.zeros_(self.gen[-1].weight)
        nn.init.zeros_(self.gen[-1].bias)

    def forward(self, base_W: torch.Tensor, prefs: torch.Tensor) -> torch.Tensor:
        """
        base_W: (d_out, d_in), prefs: (B, P)
        We average s(p) over batch to produce a single update per forward pass
        (stable and cheap). If you want per-batch-item adapters, you’d need to
        expand the matmul; this version keeps the conv weight shared inside a batch.
        """
        s = self.gen(prefs).mean(dim=0)  # (r,)
        W_update = self.A @ torch.diag(s) @ self.B  # (d_out, d_in)
        return base_W + W_update

class PrefGATConv(MessagePassing):
    def __init__(
        self,
        in_channels,                 # int or Tuple[int, int]
        out_channels: int,
        heads: int = 1,
        edge_dim: int = 0,
        pref_dim: int = 0,
        lora_rank: int = 0,          # 0 disables LoRA
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = False,  # keep False to match your original
        bias: bool = True,
        **kwargs
    ):
        super().__init__(node_dim=0, aggr='add', **kwargs)
        self.in_channels_src, self.in_channels_dst = (
            in_channels if isinstance(in_channels, (tuple, list)) else (in_channels, in_channels)
        )
        self.out_channels = out_channels
        self.heads = heads
        self.edge_dim = edge_dim
        self.pref_dim = pref_dim
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # Linear projections for src and dst (no bias; we’ll add bias later if needed)
        self.lin_src = nn.Linear(self.in_channels_src, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(self.in_channels_dst, heads * out_channels, bias=False)

        # LoRA on source projection (optional)
        self.use_lora = lora_rank > 0
        if self.use_lora:
            self.lora_src = LoRAAdapter(
                d_out=heads * out_channels,
                d_in=self.in_channels_src,
                r=lora_rank,
                pref_dim=pref_dim
            )

        # Attention parameters (a_l, a_r) per head as elementwise (dot) with projected features
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        # Optional linear for edge features into attention logit
        if edge_dim and edge_dim > 0:
            self.lin_edge = nn.Linear(edge_dim, heads, bias=False)
        else:
            self.lin_edge = None

        # Preference-conditioned attention bias
        if pref_dim and pref_dim > 0 and edge_dim and edge_dim > 0:
            self.pref_bias = PrefAttnBias(pref_dim=pref_dim, edge_dim=edge_dim, hidden=64)
        elif pref_dim and pref_dim > 0:
            # still allow bias from prefs alone
            self.pref_bias = PrefAttnBias(pref_dim=pref_dim, edge_dim=0, hidden=64)
        else:
            self.pref_bias = None

        # Output bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr=None, prefs=None, batch_src=None):
        """
        x: Tensor or (x_src, x_dst)
        edge_index: [2, E]
        edge_attr: (E, edge_dim) or None
        prefs: (B, P)   # one prefs vector per environment in batch
        batch_src: (N_src,)  # maps each src node to its batch id in [0, B-1]
        """
        H, C = self.heads, self.out_channels

        if isinstance(x, (tuple, list)):
            x_src, x_dst = x[0], x[1]
        else:
            x_src = x_dst = x

        # Project
        W_src = self.lin_src.weight  # (H*C, in_src)
        if self.use_lora and (prefs is not None):
            W_src = self.lora_src(W_src, prefs)
        h_src = F.linear(x_src, W_src)                    # (N_src, H*C)
        h_dst = F.linear(x_dst, self.lin_dst.weight)      # (N_dst, H*C)

        # reshape to heads
        h_src = h_src.view(-1, H, C)
        h_dst = h_dst.view(-1, H, C)

        # Compute attention logits a_l^T h_i + a_r^T h_j
        # Gather per-edge node projections
        src, dst = edge_index[0], edge_index[1]
        h_i = h_src[src]   # (E, H, C)
        h_j = h_dst[dst]   # (E, H, C)

        alpha = (h_i * self.att_src).sum(dim=-1) + (h_j * self.att_dst).sum(dim=-1)  # (E, H)

        # Edge feature contribution
        if self.lin_edge is not None and edge_attr is not None and edge_attr.numel() > 0:
            alpha = alpha + self.lin_edge(edge_attr)  # (E, H)

        # Preference-conditioned bias (shared across heads as a scalar added to each head)
        if self.pref_bias is not None and prefs is not None:
            if batch_src is None:
                raise ValueError("batch_src is required for preference-conditioned bias.")
            # Map each edge to the prefs of its SOURCE node's batch
            # batch_src: (N_src,), src: (E,)
            pref_per_edge = prefs[batch_src[src]]  # (E, P)
            bias_scalar = self.pref_bias(pref_per_edge, edge_attr if edge_attr is not None else torch.zeros(src.size(0), 0, device=src.device))
            alpha = alpha + bias_scalar.unsqueeze(-1)  # broadcast to heads

        # LeakyReLU, softmax over incoming edges of each dst node (per head)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, dst)  # (E, H)

        if self.dropout > 0.0 and self.training:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        # Messages: use h_i or h_j? Standard GAT uses neighbor's (source->dst) projected features as message.
        # Here edge_index is i->j with src->dst, message is from src to dst, so use h_i
        m = h_i * alpha.unsqueeze(-1)  # (E, H, C)

        # Aggregate on dst
        out = torch.zeros(h_dst.size(0), H, C, device=h_dst.device)
        out = out.index_add(0, dst, m)  # sum over edges to each dst node

        # Flatten heads
        out = out.view(-1, H * C)

        if self.bias is not None:
            out = out + self.bias

        return out

class PrefAwareHeteroConv(nn.Module):
    """
    A tiny HeteroConv-like wrapper that:
      - stores a dict: { (src, rel, dst): conv }
      - in forward(), loops relations and calls conv((x_src, x_dst), edge_index, edge_attr, **call_kwargs)
      - call_kwargs entries can be:
          * a broadcast tensor (same for all relations), OR
          * a dict keyed by relation tuples (or keyed by stringified relation keys)
      - aggregates per-dst node-type by sum (or mean if you change aggr).
    """
    def __init__(self, conv_dict, aggr="sum"):
        super().__init__()
        self.rels = list(conv_dict.keys())
        # store convs in ModuleDict with stable string keys
        self.convs = nn.ModuleDict({ self._rel_key(r): conv for r, conv in conv_dict.items() })
        assert aggr in ("sum", "mean")
        self.aggr = aggr

    @staticmethod
    def _rel_key(rel):
        # canonical string representation used for ModuleDict keys
        return "__".join(rel)

    def _get_per_relation_value(self, v, rel):
        """
        Handles:
          - broadcast tensor (return v)
          - dict keyed by tuple (return v[rel] if present)
          - dict keyed by string key (return v[self._rel_key(rel)] if present)
        """
        if isinstance(v, dict):
            # try tuple key
            if rel in v:
                return v[rel]
            # try string key
            s = self._rel_key(rel)
            if s in v:
                return v[s]
            return None
        else:
            return v  # broadcast tensor

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, **extra):
        # Collect per-dst outputs
        out_accum = {ntype: [] for ntype in x_dict.keys()}

        for rel in self.rels:
            src, _, dst = rel
            key = self._rel_key(rel)
            conv = self.convs[key]

            x_src = x_dict[src]
            x_dst = x_dict[dst]
            edge_index = edge_index_dict[rel]
            edge_attr = None
            if edge_attr_dict is not None and rel in edge_attr_dict:
                edge_attr = edge_attr_dict[rel]

            # Build call kwargs for this relation
            call_kwargs = {}
            for k, v in extra.items():
                val = self._get_per_relation_value(v, rel)
                if val is not None:
                    call_kwargs[k] = val
                # If val is None: this extra wasn't provided for this relation.
                # We intentionally do NOT insert None values, to avoid overwriting defaults.

            # If conv expects batch_src and it wasn't provided in call_kwargs,
            # we will not force anything here — PrefGATConv raises if it's required.
            # But common usage should pass batch_src_dict keyed exactly by rel.

            out = conv((x_src, x_dst), edge_index, edge_attr=edge_attr, **call_kwargs)
            out_accum[dst].append(out)

        # Aggregate per node type
        out_dict = {}
        for ntype, parts in out_accum.items():
            if len(parts) == 0:
                # No incoming messages: return a zero-like tensor with same device and width
                # We need a size -- use x_dict[ntype].shape[0] and assume same hidden dim (conv out)
                out_dict[ntype] = torch.zeros_like(x_dict[ntype])
            else:
                stacked = torch.stack(parts, dim=0)  # (n_rel, N_dst, D)
                out_dict[ntype] = stacked.sum(dim=0) if self.aggr == "sum" else stacked.mean(dim=0)

        return out_dict