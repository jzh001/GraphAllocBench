"""Prepare two per-step rollouts under two one-hot preferences, for the
side-by-side allocation animation (images/plot_dependency_network_gif_pair.py).

Defaults use one-hot preferences on Objectives 3 and 4: for problem_6a these
serve disjoint demand sets AND cleanly show preference adherence (the emphasised
objective's value dominates the objective vector J). Pass two objective indices
to override. Also records the objective vector at each step.

Requires the trained checkpoints + torch. Writes cache/scalability/pair_<prob>.npz.

    python images/gen_dependency_network_pair.py problem_6a arch-3 0
    python images/gen_dependency_network_pair.py problem_6a arch-3 0 1 3   # one-hot obj1 vs obj3
"""
import os, sys, glob, warnings
import numpy as np
warnings.filterwarnings("ignore")
from graphallocbench.city_env import CityPlannerEnv
from graphallocbench.evaluation.model_utils import load_model
from stable_baselines3.common.vec_env import DummyVecEnv

PROB = sys.argv[1] if len(sys.argv) > 1 else 'problem_6a'
ARCH = sys.argv[2] if len(sys.argv) > 2 else 'arch-3'
SEED = sys.argv[3] if len(sys.argv) > 3 else '0'
OBJ_A = int(sys.argv[4]) if len(sys.argv) > 4 else 3
OBJ_B = int(sys.argv[5]) if len(sys.argv) > 5 else 4
os.makedirs('cache/scalability', exist_ok=True)
OUT = f'cache/scalability/pair_{PROB}.npz'

mp = sorted(glob.glob(f'models/GraphAllocBench-GNN-v1-1/{ARCH}/{PROB}/{SEED}/model_step_*.zip'))[-1]
print('checkpoint:', mp, flush=True)
env = CityPlannerEnv(PROB); N, T = env.n_objectives, env.max_steps
D, R = env.demand_count, env.resource_count
req = np.zeros((D, R))
for u, v in env.config['dependencies']:
    req[u, v] = 1
S = req @ req.T; np.fill_diagonal(S, 0)

vec = DummyVecEnv([lambda: CityPlannerEnv(env.config_path)])
model = load_model(mp, env=vec)

def onehot(i):
    p = np.zeros(N); p[i] = 1.0; return p

def rollout(pref):
    renv = CityPlannerEnv(env.config_path); renv.fix_weights(pref)
    obs, _ = renv.reset(); done, t = False, 0
    sp, sa, ss, sj = [], [], [], []
    while not done and t < T:
        a, _ = model.predict(obs, deterministic=True); a = np.asarray(a).reshape(-1)[:2]
        obs, _, term, trunc, info = renv.step(a); done = term or trunc
        renv.calculate_productions()
        sp.append(renv.productions.copy()); sa.append(a.astype(int))
        ss.append(int(info.get('correct_allocation_reward', 0)))
        sj.append(np.array(renv.get_objectives(renv.productions, renv.allocation_matrix)))
        t += 1
    return np.array(sp), np.array(sa), np.array(ss), np.array(sj)

PREF_A, PREF_B = onehot(OBJ_A), onehot(OBJ_B)
spA, saA, ssA, sjA = rollout(PREF_A)
spB, saB, ssB, sjB = rollout(PREF_B)
print(f'A (obj{OBJ_A}): units={int(spA[-1].sum())} served={int((spA[-1]>0).sum())} J={np.round(sjA[-1],1).tolist()}')
print(f'B (obj{OBJ_B}): units={int(spB[-1].sum())} served={int((spB[-1]>0).sum())} J={np.round(sjB[-1],1).tolist()}')

np.savez(OUT, S=S, req=req, max_steps=T, n_objectives=N,
         pref_A=PREF_A, obj_A=OBJ_A, step_prod_A=spA, step_action_A=saA, step_success_A=ssA, step_obj_A=sjA,
         pref_B=PREF_B, obj_B=OBJ_B, step_prod_B=spB, step_action_B=saB, step_success_B=ssB, step_obj_B=sjB)
print('saved', OUT)
