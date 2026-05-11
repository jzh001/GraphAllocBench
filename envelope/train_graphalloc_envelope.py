#!/usr/bin/env python3
"""Train Envelope Q-Learning on GraphAllocBench problems."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envelope.common import (  # noqa: E402
    GraphAllocDiscreteEnv,
    GraphAllocEnvelopeCQN,
    EnvelopePolicyAdapter,
    EnvelopeTrainingConfig,
    ensure_pymoo_factory,
    ensure_numpy_copy_patch,
    ensure_numpy_float_patch,
)
from graphallocbench.evaluation import utils as eval_utils  # noqa: E402
from graphallocbench.evaluation.inference import run_experiments  # noqa: E402
from graphallocbench.evaluation.analytical import get_analytical_objectives  # noqa: E402
from graphallocbench.city_env.env_model import CityPlannerEnv  # noqa: E402

DEFAULT_ENVELOPE_ROOT = SCRIPT_DIR / "external" / "MORL"

# Cache so _create_patched_agent_class is called at most once per MetaAgent class.
_PATCHED_CLASS_CACHE: dict = {}


def _extend_sys_path(envelope_root: Path) -> None:
    source_path = envelope_root / "synthetic"
    if not envelope_root.exists():
        raise FileNotFoundError(
            f"MORL repo not found at {envelope_root}. "
            "Run: git submodule update --init --recursive"
        )
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))


def _import_envelope_modules():
    ensure_numpy_copy_patch()
    ensure_numpy_float_patch()
    ensure_pymoo_factory()
    from crl.envelope.meta import MetaAgent  # type: ignore
    from crl.envelope.models import get_new_model  # type: ignore
    return MetaAgent, get_new_model


def _patch_model_H(ModelClass) -> None:
    """Replace EnvelopeLinearCQN.H with a bool-mask version (PyTorch >= 2.0 compat).

    masked_select requires a BoolTensor in modern PyTorch; the original uses
    ByteTensor which raises a RuntimeError in torch >= 2.0.
    """
    _LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    def H_fixed(self, Q, w, s_num, w_num):
        mask_idx = torch.cat(
            [torch.arange(i, s_num * w_num + i, s_num) for i in range(s_num)]
        ).type(_LongTensor)
        reQ = Q.view(-1, self.action_size * self.reward_size)[mask_idx].view(-1, self.reward_size)
        reQ_ext = reQ.repeat(w_num, 1)
        w_ext = w.unsqueeze(2).repeat(1, self.action_size * w_num, 1).view(-1, self.reward_size)
        prod = torch.bmm(reQ_ext.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()
        prod = prod.view(-1, self.action_size * w_num)
        inds = prod.max(1)[1]
        bool_mask = torch.zeros(prod.size(), dtype=torch.bool, device=prod.device)
        bool_mask.scatter_(1, inds.unsqueeze(1), True)
        bool_mask = bool_mask.view(-1, 1).repeat(1, self.reward_size)
        return reQ_ext.masked_select(bool_mask).view(-1, self.reward_size)

    ModelClass.H = H_fixed


def _create_patched_agent_class(MetaAgent):
    """Return a MetaAgent subclass with NumPy/print compatibility fixes (cached)."""
    if MetaAgent in _PATCHED_CLASS_CACHE:
        return _PATCHED_CLASS_CACHE[MetaAgent]

    _use_cuda = torch.cuda.is_available()
    _FloatTensor = torch.cuda.FloatTensor if _use_cuda else torch.FloatTensor

    class GraphAllocEnvelopeAgent(MetaAgent):
        """MetaAgent with NumPy 1.24 and PyTorch 2.x compatibility fixes."""

        def sample(self, pop, pri, k):
            """Fixed: np.float was removed in NumPy 1.24."""
            pri = np.array(list(pri)).astype(np.float64)
            inds = np.random.choice(
                range(len(pop)), k, replace=False, p=pri / pri.sum()
            )
            return [pop[i] for i in inds]

        def memorize(self, state, action, next_state, reward, terminal):
            """Store a transition and update priority; no verbose beta printout."""
            from torch.autograd import Variable  # type: ignore

            self.trans_mem.append(self.trans(
                torch.from_numpy(state).type(_FloatTensor),
                action,
                torch.from_numpy(next_state).type(_FloatTensor),
                torch.from_numpy(reward).type(_FloatTensor),
                terminal,
            ))

            # Random preference for computing TD-error priority.
            preference = torch.randn(self.model_.reward_size)
            preference = (torch.abs(preference) / torch.norm(preference, p=1)).type(_FloatTensor)

            state_t = torch.from_numpy(state).type(_FloatTensor)
            with torch.no_grad():
                _, q = self.model_(
                    Variable(state_t.unsqueeze(0)),
                    Variable(preference.unsqueeze(0)),
                )
            q_a = q[0, action].data
            wq = float(preference.dot(q_a))
            wr = float(preference.dot(torch.from_numpy(reward).type(_FloatTensor)))

            if not terminal:
                next_state_t = torch.from_numpy(next_state).type(_FloatTensor)
                with torch.no_grad():
                    hq, _ = self.model_(
                        Variable(next_state_t.unsqueeze(0)),
                        Variable(preference.unsqueeze(0)),
                    )
                whq = float(preference.dot(hq.data[0]))
                p = abs(wr + self.gamma * whq - wq)
            else:
                self.w_kept = None
                if self.epsilon_decay:
                    self.epsilon -= self.epsilon_delta
                if self.homotopy:
                    self.beta += self.beta_delta
                    self.beta_delta = (
                        (self.beta - self.beta_init) * self.beta_expbase
                        + self.beta_init
                        - self.beta
                    )
                p = abs(wr - wq)

            self.priority_mem.append(p + 1e-5)
            if len(self.trans_mem) > self.mem_size:
                self.trans_mem.popleft()
                self.priority_mem.popleft()

    _PATCHED_CLASS_CACHE[MetaAgent] = GraphAllocEnvelopeAgent
    return GraphAllocEnvelopeAgent


class _NoOpWriter:
    def add_scalar(self, *a, **kw): pass
    def close(self): pass


def _sample_preference(rng: np.random.Generator, n_objectives: int, alpha: float) -> np.ndarray:
    pref = rng.dirichlet(np.ones(n_objectives) * alpha).astype(np.float32)
    return pref / pref.sum()


def _prepare_preferences(step: float, n_objectives: int, n_partitions: int = 12) -> np.ndarray:
    from pymoo.util.ref_dirs import get_reference_directions
    return get_reference_directions("das-dennis", n_objectives, n_partitions=n_partitions).astype(np.float32)


def _build_meta_args(args: argparse.Namespace, episode_num: int) -> SimpleNamespace:
    return SimpleNamespace(
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        episode_num=max(1, episode_num),
        mem_size=args.mem_size,
        batch_size=args.batch_size,
        weight_num=args.weight_num,
        beta=args.beta,
        homotopy=args.homotopy,
        optimizer=args.optimizer,
        lr=args.learning_rate,
        update_freq=args.update_freq,
    )


def _evaluate_agent(agent, config_path: str, preferences: np.ndarray, seed: int) -> dict:
    eval_env = CityPlannerEnv(config_path)
    adapter = EnvelopePolicyAdapter(agent, eval_env.demand_count)
    final_objectives, _ = run_experiments(
        env=eval_env,
        model=adapter,
        preferences=preferences,
        deterministic=True,
        random_seed=42,
    )
    objectives = np.asarray(final_objectives, dtype=np.float32)
    hv = float(eval_utils.calculate_hypervolume(objectives))
    ideal_objs = (
        get_analytical_objectives(env=eval_env) if eval_env.demand_count < 6 else None
    )
    ideal_hv = float(eval_utils.calculate_hypervolume(ideal_objs)) if ideal_objs is not None else 0.0
    normalized_hv = hv / ideal_hv if ideal_hv > 0 else 0.0
    pnd = float(eval_utils.calculate_non_dominated(objectives))
    ordering = float(
        eval_utils.calculate_ordering_score(
            env=eval_env, model=adapter, random_seed=42,
        )
    )
    return {
        "hypervolume": hv,
        "normalized_hypervolume": normalized_hv,
        "percent_non_dominated": pnd,
        "ordering_score": ordering,
        "objectives": objectives,
    }


def _save_checkpoint(
    agent,
    save_dir: Path,
    step: int,
    problem_name: str,
    seed: int,
    args: argparse.Namespace,
    arch: dict,
) -> Path:
    ckpt_dir = save_dir / problem_name / f"seed-{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"envelope_step_{step}.pt"
    torch.save(
        {
            "state_dict": agent.model_.state_dict(),
            "train_args": vars(args),
            "arch": arch,
        },
        path,
    )
    return path


def train_single_seed(
    args: argparse.Namespace,
    seed: int,
    MetaAgent_cls,
    get_new_model,
) -> None:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = GraphAllocDiscreteEnv(args.config)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    reward_size = env.n_objectives
    arch = {
        "state_size": state_size, "action_size": action_size, "reward_size": reward_size,
        "hidden_size": args.hidden_size, "n_layers": args.n_layers,
    }

    # Envelope uses episode-based scheduling; derive episode_num from step budget.
    episode_num = max(1, args.total_steps // env.max_episode_steps)
    meta_args = _build_meta_args(args, episode_num)

    # Fixed-width model: ~200K params regardless of problem size (matches PD-MORL).
    model = GraphAllocEnvelopeCQN(
        state_size, action_size, reward_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
    )

    AgentCls = _create_patched_agent_class(MetaAgent_cls)
    agent = AgentCls(model, meta_args, is_train=True)

    problem_name = Path(args.config).stem
    pref_grid = _prepare_preferences(args.pref_grid_step, env.n_objectives)
    _use_cuda = torch.cuda.is_available()

    if getattr(args, "save_logs", True):
        writer_dir = Path(args.log_dir) / problem_name / f"seed-{seed}"
        writer_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(writer_dir))
    else:
        writer = _NoOpWriter()

    step = 0
    start_time = time.time()

    with tqdm(total=args.total_steps, desc=f"seed {seed}", unit="step", leave=False) as pbar:
        while step < args.total_steps:
            # ── new episode ──────────────────────────────────────────────────
            pref = _sample_preference(rng, env.n_objectives, args.dirichlet_alpha)
            env.set_episode_preference(pref)
            pref_tensor = torch.as_tensor(pref, dtype=torch.float32)
            if _use_cuda:
                pref_tensor = pref_tensor.cuda()
            agent.w_kept = pref_tensor

            state = env.reset()
            terminal = False
            episode_reward = np.zeros(env.n_objectives, dtype=np.float32)
            episode_len = 0

            while not terminal and step < args.total_steps:
                action = agent.act(state, pref_tensor)
                next_state, reward_vec, terminal, _info = env.step(action)
                agent.memorize(state, action, next_state, reward_vec, terminal)
                agent.learn()

                state = next_state
                episode_reward += reward_vec
                episode_len += 1
                step += 1
                pbar.update(1)

                if step % args.eval_interval == 0:
                    metrics = _evaluate_agent(agent, args.config, pref_grid, seed)
                    writer.add_scalar("eval/hypervolume", metrics["hypervolume"], step)
                    writer.add_scalar("eval/normalized_hypervolume", metrics["normalized_hypervolume"], step)
                    writer.add_scalar("eval/percent_non_dominated", metrics["percent_non_dominated"], step)
                    writer.add_scalar("eval/ordering_score", metrics["ordering_score"], step)

                    out_dir = Path(args.save_dir) / problem_name / f"seed-{seed}"
                    out_dir.mkdir(parents=True, exist_ok=True)

                    if getattr(args, "save_checkpoints", True):
                        ckpt = _save_checkpoint(agent, Path(args.save_dir), step, problem_name, seed, args, arch)
                        ckpt_ref = str(ckpt)
                    else:
                        ckpt_ref = ""

                    serial = {
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in metrics.items()
                    }
                    (out_dir / f"metrics_step_{step}.json").write_text(
                        json.dumps({"step": step, **serial, "checkpoint": ckpt_ref}, indent=2),
                        encoding="utf-8",
                    )
                    tqdm.write(
                        f"[seed={seed} step={step}] "
                        f"HV={metrics['hypervolume']:.4f} | "
                        f"NormHV={metrics['normalized_hypervolume']:.4f} | "
                        f"PND={metrics['percent_non_dominated']:.4f} | "
                        f"Order={metrics['ordering_score']:.4f}"
                    )

            writer.add_scalar("train/episode_length", episode_len, step)
            for idx, val in enumerate(episode_reward):
                writer.add_scalar(f"train/objective_{idx}", float(val), step)

    elapsed = (time.time() - start_time) / 60
    tqdm.write(f"[seed={seed}] Training complete in {elapsed:.2f} min.")

    if getattr(args, "save_checkpoints", True):
        final = _save_checkpoint(
            agent, Path(args.save_dir), args.total_steps, problem_name, seed, args, arch
        )
        tqdm.write(f"[seed={seed}] Final checkpoint → {final}")

    writer.close()


def train(args: argparse.Namespace) -> None:
    _extend_sys_path(Path(args.envelope_root))
    MetaAgent_cls, get_new_model = _import_envelope_modules()
    seed_list = args.seeds if args.seeds else [args.seed]
    for seed in seed_list:
        print(f"Training seed {seed} on {Path(args.config).stem}")
        train_single_seed(args, seed, MetaAgent_cls, get_new_model)


def parse_args() -> argparse.Namespace:
    default_config = REPO_ROOT / "graphallocbench" / "config" / "problems" / "problem_0.yml"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=str(default_config))
    p.add_argument("--envelope-root", default=str(DEFAULT_ENVELOPE_ROOT))
    p.add_argument("--total-steps", type=int, default=EnvelopeTrainingConfig.total_steps)
    p.add_argument("--mem-size", type=int, default=EnvelopeTrainingConfig.mem_size)
    p.add_argument("--batch-size", type=int, default=EnvelopeTrainingConfig.batch_size)
    p.add_argument("--eval-interval", type=int, default=EnvelopeTrainingConfig.eval_interval)
    p.add_argument("--gamma", type=float, default=EnvelopeTrainingConfig.gamma)
    p.add_argument("--learning-rate", type=float, default=EnvelopeTrainingConfig.learning_rate)
    p.add_argument("--epsilon", type=float, default=EnvelopeTrainingConfig.epsilon)
    p.add_argument("--epsilon-decay", action="store_true", default=EnvelopeTrainingConfig.epsilon_decay)
    p.add_argument("--no-epsilon-decay", dest="epsilon_decay", action="store_false")
    p.add_argument("--weight-num", type=int, default=EnvelopeTrainingConfig.weight_num)
    p.add_argument("--beta", type=float, default=EnvelopeTrainingConfig.beta)
    p.add_argument("--homotopy", action="store_true", default=EnvelopeTrainingConfig.homotopy)
    p.add_argument("--no-homotopy", dest="homotopy", action="store_false")
    p.add_argument("--update-freq", type=int, default=EnvelopeTrainingConfig.update_freq)
    p.add_argument("--optimizer", choices=["Adam", "RMSprop"], default=EnvelopeTrainingConfig.optimizer)
    p.add_argument("--hidden-size", type=int, default=EnvelopeTrainingConfig.hidden_size,
                   help="Hidden-layer width (matches PD-MORL default of 256)")
    p.add_argument("--n-layers", type=int, default=EnvelopeTrainingConfig.n_layers,
                   help="Number of hidden layers (matches PD-MORL default of 3)")
    p.add_argument("--dirichlet-alpha", type=float, default=EnvelopeTrainingConfig.dirichlet_alpha)
    p.add_argument("--pref-grid-step", type=float, default=EnvelopeTrainingConfig.pref_grid_step)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Informational only; device is set via CUDA globals in envelope/meta.py.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--log-dir", default=str(SCRIPT_DIR / "runs"))
    p.add_argument("--save-dir", default=str(SCRIPT_DIR / "checkpoints"))
    return p.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    Path(parsed.log_dir).mkdir(parents=True, exist_ok=True)
    Path(parsed.save_dir).mkdir(parents=True, exist_ok=True)
    train(parsed)
