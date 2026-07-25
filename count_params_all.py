"""Count PPO policy parameter counts for every benchmark problem across the
three architectures reported in the paper:

    arch 0 -> MLP
    arch 2 -> HGNN + Mean/Max pooling
    arch 3 -> HGNN + (preference-conditioned) Attention pooling

For each (problem, arch) we build the exact PPO policy that training builds
(see graphallocbench/train_utils/general.py) and count policy parameters.

We report two counts for the HGNN variants:
  * full      : every parameter in model.policy
  * effective : excludes the `proj_post_gnn` and `gat_norm` modules, which are
                instantiated for checkpoint compatibility but never used in the
                forward pass. The paper's Table 1 reports the effective count.
The MLP has no unused layers, so full == effective for it.
"""
import os
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")
os.environ.setdefault("WANDB_MODE", "disabled")

from stable_baselines3 import PPO  # noqa: E402

from graphallocbench.city_env.env_model import CityPlannerEnv  # noqa: E402
from graphallocbench.city_env.architectures import architectures  # noqa: E402

PROBLEMS = [
    "0",
    "1a", "1b", "1c",
    "2a", "2b", "2c",
    "3a", "3b",
    "4a", "4b",
    "5a", "5b", "5c", "5d", "5e",
    "6a", "6b", "6c",
]

ARCHS = {0: "MLP", 2: "HGNN-MeanMax", 3: "HGNN-Attention"}


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def unused_hgnn_params(extractor) -> int:
    """Params present in the module but never used in forward()."""
    total = 0
    for attr in ("proj_post_gnn", "gat_norm"):
        m = getattr(extractor, attr, None)
        if m is not None:
            total += n_params(m)
    return total


def main() -> None:
    rows = []
    for problem in PROBLEMS:
        env = CityPlannerEnv(f"config/problems/problem_{problem}.yml")
        env.spec = SimpleNamespace(id="CityPlannerEnv-v0")
        rec = {"problem": problem}
        for arch_idx, label in ARCHS.items():
            model = PPO(
                "MultiInputPolicy", env=env, device="cpu", verbose=0,
                policy_kwargs=architectures[arch_idx], seed=0,
            )
            full = n_params(model.policy)
            if arch_idx == 0:
                eff = full
            else:
                eff = full - unused_hgnn_params(model.policy.features_extractor)
            rec[label] = (full, eff)
            del model
        rows.append(rec)
        print(
            f"problem_{problem:3s}  "
            f"MLP={rec['MLP'][1]:>9,}  "
            f"MeanMax={rec['HGNN-MeanMax'][1]:>9,} (full {rec['HGNN-MeanMax'][0]:>9,})  "
            f"Attn={rec['HGNN-Attention'][1]:>9,} (full {rec['HGNN-Attention'][0]:>9,})"
        )

    print("\n=== CSV (effective counts; matches paper Table 1) ===")
    print("problem,MLP,HGNN_MeanMax,HGNN_Attention")
    for r in rows:
        print(f"{r['problem']},{r['MLP'][1]},{r['HGNN-MeanMax'][1]},{r['HGNN-Attention'][1]}")

    print("\n=== CSV (full counts; includes unused compat layers) ===")
    print("problem,MLP,HGNN_MeanMax,HGNN_Attention")
    for r in rows:
        print(f"{r['problem']},{r['MLP'][0]},{r['HGNN-MeanMax'][0]},{r['HGNN-Attention'][0]}")


if __name__ == "__main__":
    main()
