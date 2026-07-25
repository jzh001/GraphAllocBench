"""Prepare the allocation cache for the dependency-network scalability figures.

Requires the trained GNN checkpoints (models/GraphAllocBench-GNN-v1-1) and torch.
Writes cache/scalability/alloc_<problem>.npz, consumed by:
    images/plot_dependency_network.py       (static)
    images/plot_dependency_network_gif.py   (animation)

    python images/gen_dependency_network_data.py problem_6a           # arch-0, seed 0
    python images/gen_dependency_network_data.py problem_6b arch-2

Cached arrays:
  req (D,R), S (D,D projection), demands (D,), avail (R,)
  prod_agg (D,)   production per demand summed over the preference simplex (static colour)
  step_prod (T,D) INSTANTANEOUS production per demand after each rollout step (animation)
  step_action (T,2), step_success (T,)
"""
import os, sys, glob, warnings
import numpy as np
warnings.filterwarnings("ignore")
from pymoo.util.ref_dirs import get_reference_directions
from graphallocbench.city_env import CityPlannerEnv
from graphallocbench.evaluation import run_experiments
from graphallocbench.evaluation.model_utils import load_model
from stable_baselines3.common.vec_env import DummyVecEnv

PROB = sys.argv[1] if len(sys.argv) > 1 else 'problem_6a'
ARCH = sys.argv[2] if len(sys.argv) > 2 else 'arch-0'
SEED = sys.argv[3] if len(sys.argv) > 3 else '0'
OUTDIR = 'cache/scalability'
os.makedirs(OUTDIR, exist_ok=True)
OUT = f'{OUTDIR}/alloc_{PROB}.npz'

cands = sorted(glob.glob(f'models/GraphAllocBench-GNN-v1-1/{ARCH}/{PROB}/{SEED}/model_step_*.zip'))
if not cands:
    raise SystemExit(f'no checkpoints for {ARCH}/{PROB}/{SEED}')
mp = cands[-1]                                    # latest step
print('checkpoint:', mp, flush=True)

env = CityPlannerEnv(PROB)
D, R, T = env.demand_count, env.resource_count, env.max_steps
req = np.zeros((D, R))
for u, v in env.config['dependencies']:
    req[u, v] = 1
S = req @ req.T
np.fill_diagonal(S, 0)
demands = np.array(env.config['demands'])
avail = np.array(env.config['avail_resources'])

def production_of(alloc):
    p = np.zeros(D, int)
    for d in range(D):
        m = req[d] == 1
        p[d] = alloc[d][m].min() if m.any() else 0
    return p

# (1) aggregate production over the preference simplex -> static node colour
prefs = get_reference_directions('das-dennis', env.n_objectives, n_partitions=6)
print(f'aggregating over {len(prefs)} preferences ...', flush=True)
_, allocs = run_experiments(env=env, model_path=mp, preferences=prefs, random_seed=42)
prod_agg = np.zeros(D, int)
for a in allocs:
    prod_agg += production_of(a)
print('prod_agg: served', int((prod_agg > 0).sum()), 'total', int(prod_agg.sum()))

# (2) per-step rollout under a uniform preference -> animation
print('per-step rollout ...', flush=True)
vec = DummyVecEnv([lambda: CityPlannerEnv(env.config_path)])
model = load_model(mp, env=vec)
renv = CityPlannerEnv(env.config_path)
renv.fix_weights(np.full(env.n_objectives, 1.0 / env.n_objectives))
obs, _ = renv.reset()
step_prod, step_action, step_success = [], [], []
done, t = False, 0
while not done and t < T:
    action, _ = model.predict(obs, deterministic=True)
    action = np.asarray(action).reshape(-1)[:2]
    obs, _, term, trunc, info = renv.step(action)
    done = term or trunc
    renv.calculate_productions()                       # instantaneous production
    step_prod.append(renv.productions.copy())
    step_action.append(action.astype(int))
    step_success.append(int(info.get('correct_allocation_reward', 0)))
    t += 1
step_prod = np.array(step_prod)
print('rollout: final units', int(step_prod[-1].sum()), 'served', int((step_prod[-1] > 0).sum()))

np.savez(OUT, req=req, S=S, demands=demands, avail=avail, prod_agg=prod_agg,
         step_prod=step_prod, step_action=np.array(step_action),
         step_success=np.array(step_success), n_objectives=env.n_objectives, max_steps=T)
print('saved', OUT)
