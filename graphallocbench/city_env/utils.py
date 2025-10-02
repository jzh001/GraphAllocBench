import wandb
import os
from stable_baselines3.common.callbacks import BaseCallback

class WandbTrainingCallback(BaseCallback):
    """W&B logging & periodic checkpoint callback."""
    def __init__(self, save_path = None, save_freq=50000, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_real_rewards = []
        self.episode_correct_allocation_rewards = []
        self.episode_lengths = []
        self.current_real_reward = 0
        self.current_correct_allocation_reward = 0
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        self.current_real_reward += self.locals["infos"][0].get("real_reward", 0)
        self.current_correct_allocation_reward += self.locals["infos"][0].get("correct_allocation_reward", 0)

        self.estimated_ideal_mean = self.locals["infos"][0].get("estimated_ideal_mean", 0)

        if "episode" in self.locals["infos"][0]:
            ep_info = self.locals["infos"][0]["episode"]
            self.episode_rewards.append(ep_info["r"])
            self.episode_lengths.append(ep_info["l"])
            self.episode_real_rewards.append(self.current_real_reward)
            self.episode_correct_allocation_rewards.append(self.current_correct_allocation_reward)

            real_reward_avg = (
                float(sum(self.episode_real_rewards[-100:]) / 100)
                if len(self.episode_real_rewards) > 100
                else float(sum(self.episode_real_rewards) / len(self.episode_real_rewards))
            )
            correct_allocation_reward_avg = (
                float(sum(self.episode_correct_allocation_rewards[-100:]) / 100)
                if len(self.episode_correct_allocation_rewards) > 100
                else float(sum(self.episode_correct_allocation_rewards) / len(self.episode_correct_allocation_rewards))
            )
            episode_reward_avg = (
                float(sum(self.episode_rewards[-100:]) / 100)
                if len(self.episode_rewards) > 100
                else float(sum(self.episode_rewards) / len(self.episode_rewards))
            )
            wandb.log(
                {
                    "episode_reward": float(ep_info["r"]),
                    "episode_reward_avg": episode_reward_avg,
                    "episode_length": int(ep_info["l"]),
                    "episode_ideal_mean": float(self.estimated_ideal_mean),
                    "total_correct_allocation_reward": self.current_correct_allocation_reward,
                    "correct_allocation_reward_avg": correct_allocation_reward_avg,
                }
            )
            self.current_real_reward = 0
            self.current_correct_allocation_reward = 0

        if self.save_path is not None and self.n_calls % self.save_freq == 0:
            save_file = os.path.join(self.save_path, f"model_step_{self.num_timesteps}.zip")
            self.model.save(save_file)

        return True
