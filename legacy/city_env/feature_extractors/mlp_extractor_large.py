import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MLPFeatureExtractorLarge(BaseFeaturesExtractor):
    def __init__(self, observation_space, embedding_dim=128, hidden_dim=320, n_layers=6):
        # Accept Dict observation_space (e.g., with keys 'allocation_matrix', 'prefs')
        if hasattr(observation_space, "spaces"):
            # Dict space: sum flattened dims of allocation_matrix and prefs
            obs_dim = 0
            self.obs_keys = []
            for k in ["allocation_matrix", "prefs"]:
                v = observation_space.spaces[k]
                obs_dim += int(torch.prod(torch.tensor(v.shape)))
                self.obs_keys.append(k)
            self.pref_dim = observation_space.spaces["prefs"].shape[0]
        else:
            # Fallback: flat space
            obs_dim = observation_space.shape[0]
            self.obs_keys = None
            self.pref_dim = None
        super().__init__(observation_space, features_dim=embedding_dim)

        self.n_layers = n_layers
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Main MLP layers
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        for i in range(n_layers):
            in_dim = obs_dim if i == 0 else hidden_dim
            self.linears.append(nn.Linear(in_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.acts.append(nn.SiLU())

        self.final_linear = nn.Linear(hidden_dim, embedding_dim)

        # Skip connection projection for input to hidden_dim
        self.input_proj = nn.Linear(obs_dim, hidden_dim)

        # FiLM (Feature-wise Linear Modulation) for preference conditioning
        if self.pref_dim is not None:
            self.film = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.pref_dim, hidden_dim * 2)
                ) for _ in range(n_layers)
            ])
        else:
            self.film = None

    def forward(self, observations) -> torch.Tensor:
        # If input is a dict, flatten and concatenate allocation_matrix and prefs
        if isinstance(observations, dict) or (hasattr(observations, 'keys') and 'allocation_matrix' in observations):
            flat = []
            for k in self.obs_keys:
                v = observations[k]
                flat.append(v.flatten(start_dim=1) if v.ndim > 1 else v)
            x = torch.cat(flat, dim=1)
            prefs = observations["prefs"]
        else:
            x = observations
            prefs = None

        h = x
        input_proj = self.input_proj(x)
        for i in range(self.n_layers):
            h_in = h
            h = self.linears[i](h)
            h = self.norms[i](h)
            # FiLM conditioning
            if self.film is not None and prefs is not None:
                gamma_beta = self.film[i](prefs)
                gamma, beta = gamma_beta.chunk(2, dim=-1)
                h = gamma * h + beta
            h = self.acts[i](h)
            # Skip connection from input
            h = h + input_proj
        out = self.final_linear(h)
        return out