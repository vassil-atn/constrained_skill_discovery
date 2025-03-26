from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

if TYPE_CHECKING:
    pass


@height_field_to_mesh
def maze_terrain(difficulty: float, cfg: HfMazeTerrainCfg) -> np.ndarray:
    """Generate a maze terrain with corridors at least 1m wide.

    Args:
        difficulty: The difficulty of the terrain (0 to 1). Higher values create more complex mazes.
        cfg: The configuration for the maze terrain.

    Returns:
        The height field of the terrain as a 2D numpy array.
    """
    # Convert to discrete units
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    wall_height = int(cfg.wall_height / cfg.vertical_scale)
    corridor_width_pixels = max(
        int(cfg.corridor_width / cfg.horizontal_scale), 10
    )  # Ensure minimum 1m width
    entrance_pixels = int(cfg.entrance_size / cfg.horizontal_scale)

    # Calculate maze grid dimensions
    cell_size = corridor_width_pixels * 2  # Each cell includes walls
    grid_width = width_pixels // cell_size
    grid_length = length_pixels // cell_size

    def create_maze(width: int, length: int) -> np.ndarray:
        """Create a maze using recursive backtracking algorithm."""
        # Initialize the maze grid
        maze = np.ones((width * 2 + 1, length * 2 + 1), dtype=np.int16)

        def carve_path(x: int, y: int, visited: List[Tuple[int, int]]):
            visited.append((x, y))
            maze[x * 2 + 1, y * 2 + 1] = 0  # Mark current cell as path

            # Define possible directions (right, down, left, up)
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            # np.random.shuffle(directions)
            directions = [(-1, 0), (0, 1), (0, -1), (1, 0)]

            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                if (
                    0 <= new_x < width
                    and 0 <= new_y < length
                    and (new_x, new_y) not in visited
                ):
                    # Carve path between cells
                    maze[x * 2 + 1 + dx, y * 2 + 1 + dy] = 0
                    carve_path(new_x, new_y, visited)

        # Start maze generation from random position
        start_x = 0  # np.random.randint(0, width)
        start_y = 0  # np.random.randint(0, length)
        carve_path(start_x, start_y, [])
        return maze

    # Generate base maze
    maze_grid = create_maze(grid_width, grid_length)

    # Scale up the maze to full resolution
    hf_raw = np.ones((width_pixels, length_pixels)) * wall_height

    # Convert maze grid to height field
    for i in range(maze_grid.shape[0]):
        for j in range(maze_grid.shape[1]):
            if maze_grid[i, j] == 0:  # Path
                x_start = i * corridor_width_pixels
                y_start = j * corridor_width_pixels
                x_end = min(x_start + corridor_width_pixels, width_pixels)
                y_end = min(y_start + corridor_width_pixels, length_pixels)
                hf_raw[x_start:x_end, y_start:y_end] = 0

    # Create entrance and exit
    # Entrance at the bottom
    hf_raw[
        0:entrance_pixels,
        length_pixels // 2 - entrance_pixels // 2 : length_pixels // 2
        + entrance_pixels // 2,
    ] = 0
    # Exit at the top
    hf_raw[
        -entrance_pixels:,
        length_pixels // 2 - entrance_pixels // 2 : length_pixels // 2
        + entrance_pixels // 2,
    ] = 0

    # create a flat platform at the center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)
    # get the height of the platform at the corner of the platform
    x_pf = width_pixels // 2 - platform_width
    y_pf = length_pixels // 2 - platform_width
    hf_raw[x_pf : x_pf + 2 * platform_width, y_pf : y_pf + 2 * platform_width] = 0
    # Add difficulty-based features
    # if difficulty > 0.5:
    #     # Add some random dead ends
    #     num_dead_ends = int((difficulty - 0.5) * 20)
    #     for _ in range(num_dead_ends):
    #         x = np.random.randint(corridor_width_pixels, width_pixels - corridor_width_pixels)
    #         y = np.random.randint(corridor_width_pixels, length_pixels - corridor_width_pixels)
    #         if hf_raw[x, y] == wall_height:
    #             hf_raw[x:x+corridor_width_pixels, y:y+corridor_width_pixels] = 0

    return np.rint(hf_raw).astype(np.int16)


@configclass
class HfMazeTerrainCfg(HfTerrainBaseCfg):
    """Configuration for maze terrain generation."""

    function = maze_terrain

    wall_height: float = 1.0  # Height of maze walls in meters
    corridor_width: float = 1.0  # Width of corridors in meters
    entrance_size: float = 2.0  # Size of entrance/exit in meters
    platform_width: float = 1.0  # Width of the platform in meters
