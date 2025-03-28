from typing import Any, Tuple

import torch

from skrl.envs.wrappers.torch.base import Wrapper


class IsaacOrbitWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Orbit environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Orbit environment
        """
        super().__init__(env)

        self._reset_once = True
        self._obs_dict = None

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        self._obs_dict, rewards, terminated, truncated, info = self._env.step(actions)

        # if len(reward_dict.keys()) == 1:
        #     return self._obs_dict, reward_dict[next(iter(reward_dict))].view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), info

        return (
            self._obs_dict,
            rewards.view(-1, 1),
            terminated.view(-1, 1),
            truncated.view(-1, 1),
            info,
        )

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        if self._reset_once:
            self._obs_dict, info = self._env.reset()
            self._reset_once = False
        return self._obs_dict, info

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        pass

    def close(self) -> None:
        """Close the environment"""
        self._env.close()
