# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import builtins

import carb
import torch
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.envs.manager_based_env_cfg import ManagerBasedEnvCfg

from isaaclab.managers import EventManager
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.timer import Timer
import omni.log

# from .action_group_base_env import ActionGroupBaseEnv


class SkillDiscoveryBaseEnv(ManagerBasedEnv):
    """Base environment for contact tracking tasks. It adds some additional tensors"""

    def __init__(self, cfg: ManagerBasedEnvCfg):
        """Initialize the environment.

        Args:
            cfg: The configuration object for the environment.

        Raises:
            RuntimeError: If a simulation context already exists. The environment must always create one
                since it configures the simulation context and controls the simulation.
        """
        # check that the config is valid
        cfg.validate()
        # store inputs to class
        self.cfg = cfg
        # initialize internal variables
        self._is_closed = False

        # set the seed for the environment
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        else:
            omni.log.warn("Seed not set for the environment. The environment creation may not be deterministic.")

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            # the type-annotation is required to avoid a type-checking error
            # since it gets confused with Isaac Sim's SimulationContext class
            self.sim: SimulationContext = SimulationContext(self.cfg.sim)
        else:
            # simulation context should only be created before the environment
            # when in extension mode
            if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
                raise RuntimeError("Simulation context already exists. Cannot create a new one.")
            self.sim: SimulationContext = SimulationContext.instance()

        # make sure torch is running on the correct device
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tEnvironment seed      : {self.cfg.seed}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}")
        print(f"\tEnvironment step-size : {self.step_dt}")

        if self.cfg.sim.render_interval < self.cfg.decimation:
            msg = (
                f"The render interval ({self.cfg.sim.render_interval}) is smaller than the decimation "
                f"({self.cfg.decimation}). Multiple render calls will happen for each environment step. "
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            omni.log.warn(msg)

        # counter for simulation steps
        self._sim_step_counter = 0

        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            self.scene = InteractiveScene(self.cfg.scene)
        print("[INFO]: Scene manager: ", self.scene)

        # set up camera viewport controller
        # viewport is not available in other rendering modes so the function will throw a warning
        # FIXME: This needs to be fixed in the future when we unify the UI functionalities even for
        # non-rendering modes.
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, self.cfg.viewer)
        else:
            self.viewport_camera_controller = None

        # create event manager
        # note: this is needed here (rather than after simulation play) to allow USD-related randomization events
        #   that must happen before the simulation starts. Example: randomizing mesh scale
        self.event_manager = EventManager(self.cfg.events, self)
        print("[INFO] Event Manager: ", self.event_manager)

        # apply USD-related randomization events
        if "prestartup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="prestartup")

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                self.sim.reset()
                # update scene to pre populate data buffers for assets and sensors.
                # this is needed for the observation manager to get valid tensors for initialization.
                # this shouldn't cause an issue since later on, users do a reset over all the environments so the lazy buffers would be reset.
                self.scene.update(dt=self.physics_dt)
            # add timeline event to load managers
            self.load_managers()

        # extend UI elements
        # we need to do this here after all the managers are initialized
        # this is because they dictate the sensors and commands right now
        if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
            # setup live visualizers
            self.setup_manager_visualizers()
            self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        else:
            # if no window, then we don't need to store the window
            self._window = None

        # allocate dictionary to store metrics
        self.extras = {}

        # initialize observation buffers
        self.obs_buf = {}
        self._init_buffers()

    """
    Operations - Setup.
    """

    def _init_buffers(self):
        """Initialize the extra buffers for the environment."""
        self.initial_root_states = torch.zeros(
            (self.num_envs, 9), device=self.device, dtype=torch.float
        )
        self.reset_to_evaluate = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
