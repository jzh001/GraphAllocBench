import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MLPFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embedding_dim=128, hidden_dim=128):
        # Accept Dict observation_space (e.g., with keys 'allocation_matrix', 'prefs')
        if hasattr(observation_space, "spaces"):
            # Dict space: sum flattened dims of allocation_matrix and prefs
            obs_dim = 0
            self.obs_keys = []
            for k in ["allocation_matrix", "prefs"]:
                v = observation_space.spaces[k]
                obs_dim += int(torch.prod(torch.tensor(v.shape)))
                self.obs_keys.append(k)
        else:
            # Fallback: flat space
            obs_dim = observation_space.shape[0]
            self.obs_keys = None
        super().__init__(observation_space, features_dim=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.obs_dim = obs_dim

    def forward(self, observations) -> torch.Tensor:
        # If input is a dict, flatten and concatenate allocation_matrix and prefs
        if isinstance(observations, dict) or (hasattr(observations, 'keys') and 'allocation_matrix' in observations):
            flat = []
            for k in self.obs_keys:
                v = observations[k]
                # v: (batch, ...) or (batch, n) or (batch,)
                flat.append(v.flatten(start_dim=1) if v.ndim > 1 else v)
            x = torch.cat(flat, dim=1)
        else:
            x = observations
        return self.mlp(x)