"""
Quick script to count parameters for selected architectures using PPO.

Usage: run this file from the repository root.
"""
import traceback
import torch

def param_count(module):
	return sum(p.numel() for p in module.parameters())

def try_count():
	try:
		from city_env.env_model import ResourceManagementEnv
		from city_env.feature_extractors.mlp_extractor import MLPFeatureExtractor
		from city_env.feature_extractors.hgnn_sparse_extractor import HGNNSparseExtractor
		from stable_baselines3 import PPO
	except Exception as e:
		print("Import error during setup:\n", e)
		traceback.print_exc()
		return

	config_path = "config/problems/problem_6a.yml"
	env = ResourceManagementEnv(config_path)

	archs = [
		dict(name="mlp", features_extractor_class=MLPFeatureExtractor, features_extractor_kwargs=dict(embedding_dim=128), net_arch=dict(pi=[256,256], vf=[256,256,128])),
		dict(name="hgnn_mean_max", features_extractor_class=HGNNSparseExtractor, features_extractor_kwargs=dict(features_dim=128, pooling_types=['mean','max'], pooling_strategy='concat'), net_arch=dict(pi=[256,256], vf=[256,256,128])),
		dict(name="hgnn_mha", features_extractor_class=HGNNSparseExtractor, features_extractor_kwargs=dict(features_dim=128, pooling_types=['multi_head_attention'], pooling_strategy='concat'), net_arch=dict(pi=[256,256], vf=[256,256,128])),
	]

	for a in archs:
		print("\n===", a['name'], "===")
		# extractor-only
		try:
			ext = a['features_extractor_class'](env.observation_space, **a['features_extractor_kwargs'])
			print("extractor params:", param_count(ext))
		except Exception as e:
			print("Failed to instantiate extractor:", e)
			traceback.print_exc()

		# full PPO model (policy + value nets + extractor)
		try:
			policy_kwargs = dict(features_extractor_class=a['features_extractor_class'], features_extractor_kwargs=a['features_extractor_kwargs'], net_arch=a['net_arch'])
			model = PPO("MultiInputPolicy", env, verbose=0, policy_kwargs=policy_kwargs)
			total = param_count(model.policy)
			print("PPO policy total params:", total)
		except Exception as e:
			print("Failed to build PPO model:", e)
			traceback.print_exc()

if __name__ == '__main__':
	try_count()

