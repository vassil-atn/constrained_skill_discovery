# Copyright (c) 2022-2024, Vassil Atanassov.
# All rights reserved.
#

"""Configuration for custom terrains."""

import constrained_skill_discovery.lab.terrains as csd_terrain_gen
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

FLAT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.0, 0.0), noise_step=0.02, border_width=0.25
        )
    },
)
"""Rough terrains configuration."""

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.0, 0.03), noise_step=0.02, border_width=0.25
        )
    },
)
"""Rough terrains configuration."""


MAZE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "maze": csd_terrain_gen.HfMazeTerrainCfg(
            proportion=1.0,
            wall_height=0.5,
            corridor_width=1.5,
            entrance_size=1.0,
            platform_width=2.0,
        )
    },
)
