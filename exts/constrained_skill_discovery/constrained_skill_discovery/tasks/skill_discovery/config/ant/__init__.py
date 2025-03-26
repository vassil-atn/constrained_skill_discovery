# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment (similar to OpenAI Gym Ant-v2).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Ant-SD",
    entry_point="constrained_skill_discovery.lab.envs:SkillDiscoveryTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_skill_disc_env_cfg:AntSDEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_hierarchical_cfg.yaml",
    },
)

gym.register(
    id="Ant-SD-play",
    entry_point="constrained_skill_discovery.lab.envs:SkillDiscoveryTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_skill_disc_env_cfg:AntSDEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_hierarchical_cfg.yaml",
    },
)
