from .adversarial import detection_evasion_reward
from .env import BotEnv, EnvStep, RandomEnv
from .ppo import PPOConfig, PPOTrainer, RolloutStats, train_rl
from .rewards import (
    compose_rewards,
    pixel_diff_reward,
    region_brightness_reward,
    template_match_reward,
)

__all__ = [
    "BotEnv",
    "EnvStep",
    "RandomEnv",
    "PPOConfig",
    "PPOTrainer",
    "RolloutStats",
    "train_rl",
    "compose_rewards",
    "detection_evasion_reward",
    "pixel_diff_reward",
    "region_brightness_reward",
    "template_match_reward",
]
