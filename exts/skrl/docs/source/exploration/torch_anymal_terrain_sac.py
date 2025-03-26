import isaacgym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.exploration.sac import SACLBS, SAC_DEFAULT_CONFIG
from skrl.agents.torch.exploration.sac_lds import LSDSAC, SAC_DEFAULT_CONFIG

from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.lbs_exploration.models import *
import wandb
from skrl.agents.torch.exploration.skill_discovery import TrajectoryEncoder
from skrl.agents.torch.exploration.torch_ant_sac_config import sweep_config

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed


# define models (stochastic and deterministic models) using mixins
def main(mode="train"):
    class StochasticActor(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, hidden_dims, device, clip_actions=False,
                    clip_log_std=True, min_log_std=-20, max_log_std=2):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std,squashedNormal=True)

            net_layers = []
            net_layers.append(nn.Linear(self.num_observations, hidden_dims[0]))
            net_layers.append(nn.ELU())
            
            for i in range(1, len(hidden_dims)):
                net_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
                net_layers.append(nn.ELU())
            
            net_layers.append(nn.Linear(hidden_dims[-1], 2*self.num_actions))

            self.net = nn.Sequential(*net_layers)

        def compute(self, inputs, role):
            output = self.net(inputs["states"])
            mean_value = output[:, :self.num_actions]
            log_std_value = output[:, self.num_actions:]
            return mean_value, log_std_value, {}
            # return self.mean_layer(self.net(inputs["states"])), self.log_std_layer(self.net(inputs["states"])), {}

    class Critic(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, hidden_dims, device, clip_actions=False, use_layer_norm=False):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)

            net_layers = []
            net_layers.append(nn.Linear(self.num_observations + self.num_actions, hidden_dims[0]))
            net_layers.append(nn.ELU())
            if use_layer_norm:
                net_layers.append(nn.LayerNorm(hidden_dims[0]))
            
            for i in range(1, len(hidden_dims)):
                net_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
                if use_layer_norm:
                    net_layers.append(nn.LayerNorm(hidden_dims[i]))
                net_layers.append(nn.ELU())
            
            net_layers.append(nn.Linear(hidden_dims[-1], 1))

            self.net = nn.Sequential(*net_layers)

            # self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
            #                         nn.ELU(),
            #                         nn.LayerNorm(512),
            #                         nn.Linear(512, 256),
            #                         nn.ELU(),
            #                         nn.LayerNorm(256),
            #                         nn.Linear(256, 128),
            #                         nn.ELU(),
            #                         nn.LayerNorm(128),
            #                         nn.Linear(128, 1))




        def compute(self, inputs, role):
            return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}



    if mode == "train":
        headless = True
        num_envs = 4096
    else:
        headless = False
        num_envs = 100

    # load and wrap the Isaac Gym environment
    env = load_isaacgym_env_preview4(task_name="AnymalTerrainPathfindingSkills", num_envs=num_envs,headless=headless)
    env = wrap_env(env)

    device = env.device


    # instantiate a memory as experience replay
    memory = RandomMemory(memory_size=5, num_envs=env.num_envs, device=device)


    # instantiate the agent's models (function approximators).
    # SAC requires 5 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
    hidden_dims = [512, 256, 128]
    use_layer_norm = False
    models = {}
    models["policy"] = StochasticActor(env.observation_space, env.action_space, hidden_dims,device)
    models["critic_1"] = Critic(env.observation_space, env.action_space, hidden_dims,device,use_layer_norm=use_layer_norm)
    models["critic_2"] = Critic(env.observation_space, env.action_space, hidden_dims,device,use_layer_norm=use_layer_norm)
    models["target_critic_1"] = Critic(env.observation_space, env.action_space, hidden_dims,device,use_layer_norm=use_layer_norm)
    models["target_critic_2"] = Critic(env.observation_space, env.action_space, hidden_dims,device,use_layer_norm=use_layer_norm)


    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg["num_envs"] = env.num_envs
    cfg["gradient_steps"] = 8
    cfg["batch_size"] = 4096
    cfg["discount_factor"] = 0.99
    cfg["polyak"] = 0.005
    cfg["actor_learning_rate"] = 1e-4
    cfg["critic_learning_rate"] = 1e-4
    cfg["random_timesteps"] = 32
    cfg["learning_starts"] = 32
    cfg["grad_norm_clip"] = 0.5
    cfg["learn_entropy"] = False
    cfg["entropy_learning_rate"] = 5e-3
    cfg["initial_entropy_value"] = 0.003
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["discriminator_learning_rate"] = 3e-4
    cfg["discriminator_batch_size"] = 128
    cfg["use_nstep_return"] = False
    cfg["num_envs"] = env.num_envs
    cfg["n_step"] = 1

    cfg["use_curiosity"] = False
    cfg["curiosity_lr"] = 5e-4
    cfg["curiosity_scale"] = 0.1
    cfg["anneal_curiosity"] = False
    cfg["anneal_curiosity_warmup"] = 8000
    cfg["curiosity_observation_type"] = "debugLSD"
    cfg["curiosity_beta"] = 0.1
    cfg["curiosity_epochs"] = 5
    cfg["curiosity_mini_batches"] = 12
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 2
    cfg["experiment"]["checkpoint_interval"] = 500
    cfg["experiment"]["directory"] = "runs/torch/AnymalTerrainPathfinding"
    cfg["experiment"]["wandb"] = True           # whether to use Weights & Biases
    cfg["experiment"]["wandb_kwargs"] = {"project":"AnymalPPO",   
                            "mode":"online"}       

    cfg["use_curiosity"] = False
    cfg["curiosity_lr"] = 5e-3
    cfg["curiosity_scale"] = 0.05
    cfg["anneal_curiosity"] = False
    cfg["anneal_curiosity_warmup"] = 8000
    cfg["curiosity_observation_type"] = "velocities"


    if cfg["curiosity_observation_type"] == "full_state":
        curiosity_obs_space = env.observation_space.shape[0]
        state_dim = 64
    elif cfg["curiosity_observation_type"] == "velocities":
        curiosity_obs_space = 3
        state_dim = curiosity_obs_space

    elif cfg["curiosity_observation_type"] == "positions":
        curiosity_obs_space = 3
        state_dim = curiosity_obs_space

    elif cfg["curiosity_observation_type"] == "pos_vel":
        curiosity_obs_space = 6
        state_dim = curiosity_obs_space

    elif cfg["curiosity_observation_type"] == "xvel_yaw":
        curiosity_obs_space = 2
        state_dim = curiosity_obs_space

    elif cfg["curiosity_observation_type"] == "debugLSD":
        curiosity_obs_space = env.observation_space.shape[0] - env.skill_dim # remove the skills (last two elements)
        state_dim = curiosity_obs_space

    cfg["skill_discovery"] = True
    cfg["skill_space"] = env.skill_dim
    cfg["discrete_skills"] = env.discrete_skills
    cfg["mask_xy_pos"] = False
    cfg["skill_discovery_grad_steps"] = 4
    cfg["pretrain_normalizers"] = False

    if cfg["skill_discovery"]:
        cfg["state_preprocessor_kwargs"] = {"size": env.observation_space.shape[0] - env.skill_dim, "device": device}

    if cfg["skill_discovery"]:
        cfg["lsd_model"] = TrajectoryEncoder(curiosity_obs_space, [512,512], cfg["skill_space"], device=device)
        cfg["lsd_lr"] = 1e-4


    cfg["curiosity_preprocessor"] = RunningStandardScaler
    cfg["curiosity_preprocessor_kwargs"] = {"size": curiosity_obs_space, "device": device}


    cfg["limit_curiosity_obs"] = 3.14*torch.ones(curiosity_obs_space)
    # cfg["curiosity_obs_indices"] = curiosity_obs_indices
    state_dim = curiosity_obs_space
    hidden_dim = 128
    cfg["curiosity_model"]= LBS(curiosity_obs_space,env.action_space.shape[0],state_dim,hidden_dim,device=device)


    if mode == "train":


        # logging to TensorBoard and write checkpoints (in timesteps)
        cfg["experiment"]["write_interval"] = 1
        cfg["experiment"]["checkpoint_interval"] = 500
        cfg["experiment"]["directory"] = "runs/torch/Ant"
        cfg["experiment"]["wandb"] = True           # whether to use Weights & Biases
        cfg["experiment"]["wandb_kwargs"] = {"project":"Lipschitz Skill Learning",   
                                "mode":"online"}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)


        # agent = ExplorationSAC(models=models,
        #             memory=memory,
        #             cfg=cfg,
        #             observation_space=env.observation_space,
        #             action_space=env.action_space,
        #             skill_state_space = skill_state_space,
        #             skill_space = skill_space,
        #             device=device)
        agent = LSDSAC(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

        # configure and instantiate the RL trainer
        cfg_trainer = {"timesteps": 30000, "headless": True}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

        trainer.train()

    elif mode == "eval":

        # logging to TensorBoard and write checkpoints (in timesteps)
        cfg["experiment"]["write_interval"] = 0
        cfg["experiment"]["checkpoint_interval"] = 0
        cfg["experiment"]["directory"] = "runs/torch/Ant"


        cfg["experiment"]["write_interval"] = 2
        cfg["experiment"]["checkpoint_interval"] = 500
        cfg["experiment"]["directory"] = "runs/torch/Ant"
        cfg["experiment"]["wandb"] = False           # whether to use Weights & Biases
        cfg["experiment"]["wandb_kwargs"] = {"project":"Lipschitz Skill Learning",   
                                "mode":"online"}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)

        # agent = ExplorationSAC(models=models,
        #             memory=memory,
        #             cfg=cfg,
        #             observation_space=env.observation_space,
        #             action_space=env.action_space,
        #             skill_state_space = skill_state_space,
        #             skill_space = skill_space,
        #             device=device)
        agent = LSDSAC(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)
        


        # configure and instantiate the RL trainer
        cfg_trainer = {"timesteps": 2000, "headless": True}

        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

        # download the trained agent's checkpoint from Hugging Face Hub and load it
        # path = "runs/torch/AnymalTerrain/24-01-31_11-47-24-196508_SAC/checkpoints/" + "agent_24000.pt"
        # path = "runs/torch/AnymalTerrain/24-01-31_15-23-40-830333_SACLBS/checkpoints/" + "agent_24000.pt"
        path = "runs/torch/AnymalTerrainPathfinding/24-02-01_11-54-06-548496_SACLBS/checkpoints/" + "agent_8000.pt"
        path = "runs/torch/AnymalTerrainPathfinding/24-02-01_12-17-30-146928_SACLBS/checkpoints/" + "agent_8000.pt"
        

        agent.load(path)

        # # start evaluation
        trainer.eval()

if __name__ == "__main__":
    main("train")
    # main("eval")