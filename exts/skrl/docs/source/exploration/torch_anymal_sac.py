import isaacgym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed


# define models (stochastic and deterministic models) using mixins
def main(mode="train"):
    class StochasticActor(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False,
                    clip_log_std=True, min_log_std=-5, max_log_std=2):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

            self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                    nn.ELU(),
                                    nn.Linear(512, 256),
                                    nn.ELU(),
                                    nn.Linear(256, 128),
                                    nn.ELU(),
                                    nn.Linear(128, 2*self.num_actions))

            # self.mean_layer = nn.Linear(128, self.num_actions)
            # self.log_std_layer = nn.Linear(128, self.num_actions)
            # self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        def compute(self, inputs, role):
            output = self.net(inputs["states"])
            mean_value = output[:, :self.num_actions]
            log_std_value = output[:, self.num_actions:]
            return mean_value, log_std_value, {}
            # return self.mean_layer(self.net(inputs["states"])), self.log_std_layer(self.net(inputs["states"])), {}

    class Critic(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)
            self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                    nn.ELU(),
                                    nn.Linear(512, 256),
                                    nn.ELU(),
                                    nn.Linear(256, 128),
                                    nn.ELU(),
                                    nn.Linear(128, 1))

        def compute(self, inputs, role):
            return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}



    if mode == "train":
        headless = True
        num_envs = 4096
    else:
        headless = False
        num_envs = 50

    # load and wrap the Isaac Gym environment
    env = load_isaacgym_env_preview4(task_name="Anymal", num_envs=num_envs,headless=headless)
    env = wrap_env(env)

    device = env.device


    # instantiate a memory as experience replay
    memory = RandomMemory(memory_size=2000, num_envs=env.num_envs, device=device)


    # instantiate the agent's models (function approximators).
    # SAC requires 5 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
    models = {}
    models["policy"] = StochasticActor(env.observation_space, env.action_space, device)
    models["critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["critic_2"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)


    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg["num_envs"] = env.num_envs
    cfg["gradient_steps"] = 8
    cfg["batch_size"] = 8192
    cfg["discount_factor"] = 0.99
    cfg["polyak"] = 0.05
    cfg["actor_learning_rate"] = 5e-4
    cfg["critic_learning_rate"] = 5e-4
    cfg["random_timesteps"] = 32
    cfg["learning_starts"] = 32
    cfg["grad_norm_clip"] = 0.5
    cfg["learn_entropy"] = True
    cfg["entropy_learning_rate"] = 5e-3
    cfg["initial_entropy_value"] = 1.0
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["n_step"] = 3
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 2
    cfg["experiment"]["checkpoint_interval"] = 500
    cfg["experiment"]["directory"] = "runs/torch/Anymal"
    cfg["experiment"]["wandb"] = True           # whether to use Weights & Biases
    cfg["experiment"]["wandb_kwargs"] = {"project":"AnymalPPO",   
                            "mode":"online"}       

    agent = SAC(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)


    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 24000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    if mode == "train":
        # start training
        trainer.train()
        
    else:
            # logging to TensorBoard and write checkpoints (in timesteps)
        cfg["experiment"]["write_interval"] = 0
        cfg["experiment"]["checkpoint_interval"] = 0
        cfg["experiment"]["directory"] = "runs/torch/Anymal"
        cfg["experiment"]["wandb"] = False           # whether to use Weights & Biases
        cfg["experiment"]["wandb_kwargs"] = {"project":"AnymalPPO",   
                                "mode":"online"}       

        agent = SAC(models=models,
                    memory=memory,
                    cfg=cfg,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=device)
        cfg_trainer = {"timesteps": 12000, "headless": True}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

        # download the trained agent's checkpoint from Hugging Face Hub and load it
        path = "runs/torch/Anymal/24-01-19_16-41-13-794633_SAC/checkpoints/" + "best_agent.pt"
        agent.load(path)

        # # start evaluation
        trainer.eval()

if __name__ == "__main__":
    # main("train")
    main("eval")