# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import torch
from isaaclab.envs.manager_based_env import VecEnvObs
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

from .skill_discovery_base_env import SkillDiscoveryBaseEnv

VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]
"""The environment signals processed at the end of each step.

The tuple contains batched information for each sub-environment. The information is stored in the following order:

1. **Observations**: The observations from the environment.
2. **Rewards**: The rewards from the environment.
3. **Terminated Dones**: Whether the environment reached a terminal state, such as task success or robot falling etc.
4. **Timeout Dones**: Whether the environment reached a timeout state, such as end of max episode length.
5. **Extras**: A dictionary containing additional information from the environment.
"""


class SkillDiscoveryTaskEnv(ManagerBasedRLEnv, SkillDiscoveryBaseEnv):
    pass
