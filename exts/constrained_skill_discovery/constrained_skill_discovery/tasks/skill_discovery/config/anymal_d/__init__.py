# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import (
    agents,
    hierarchical_skill_disc_env_cfg,
    reward_dense_baseline_env_cfg,
)

##
# Register Gym environments.
##
gym.register(
    id="Anymal-Skill-Discovery-D-Hierarchical",
    entry_point="constrained_skill_discovery.lab.envs:SkillDiscoveryTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": hierarchical_skill_disc_env_cfg.AnymalDHierarchicalEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_hierarchical_cfg.yaml",
    },
)

gym.register(
    id="Anymal-Skill-Discovery-D-Hierarchical-PLAY",
    entry_point="constrained_skill_discovery.lab.envs:SkillDiscoveryTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": hierarchical_skill_disc_env_cfg.AnymalDHierarchicalEnvCfg_PLAY,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_hierarchical_cfg.yaml",
    },
)

gym.register(
    id="Anymal-D-reward-baseline",
    entry_point="constrained_skill_discovery.lab.envs:SkillDiscoveryTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": reward_dense_baseline_env_cfg.AnymalDRewardDenseEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_reward_dense_cfg.yaml",
    },
)
gym.register(
    id="Anymal-D-reward-baseline-PLAY",
    entry_point="constrained_skill_discovery.lab.envs:SkillDiscoveryTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": reward_dense_baseline_env_cfg.AnymalDRewardDenseEnvCfg_PLAY,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_reward_dense_cfg.yaml",
    },
)