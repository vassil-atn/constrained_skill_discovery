from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .skill_commands import SkillCommandAction, SkillCommands


@configclass
class SkillCommandCfg(CommandTermCfg):
    """Configuration for the terrain-based position command generator."""

    class_type: type = SkillCommands

    asset_name: str = MISSING

    discrete_skills: bool = MISSING

    skill_dim: int = MISSING

    max_magnitude: float = MISSING

    normalise: bool = MISSING


@configclass
class SkillCommandActionCfg(ActionTermCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = SkillCommandAction

    discrete_skills: bool = MISSING

    skill_dim: int = MISSING
