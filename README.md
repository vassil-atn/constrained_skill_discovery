# Constrained Skill Discovery IsaacLab Implementation

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-1.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.


- Using a python interpreter that has Isaac Lab installed, install the libraries

```bash
python -m pip install -e exts/constrained_skill_discovery
```

```bash
python -m pip install -e exts/skrl
```


### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory. The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse. This helps in indexing all the python modules for intelligent suggestions while writing code.

## Docker setup

### Building Isaac Lab Base Image

Currently, we don't have the Docker for Isaac Lab publicly available. Hence, you'd need to build the docker image
for Isaac Lab locally by following the steps [here](https://isaac-sim.github.io/IsaacLab/source/deployment/index.html).

Once you have built the base Isaac Lab image, you can check it exists by doing:

```bash
docker images

# Output should look something like:
#
# REPOSITORY                       TAG       IMAGE ID       CREATED          SIZE
# isaac-lab-base                   latest    28be62af627e   32 minutes ago   18.9GB
```

### Building Constrained Skill Discovery Image

Following above, you can build the docker container for this project. It is called `constrained-skill-discovery-lab`. However,
you can modify this name inside the [`docker/docker-compose.yaml`](docker/docker-compose.yaml).

```bash
cd docker
docker compose --env-file .env.base --file docker-compose.yaml build constrained-skill-discovery-lab
```

You can verify the image is built successfully using the same command as earlier:

```bash
docker images

# Output should look something like:
#
# REPOSITORY                       TAG       IMAGE ID       CREATED             SIZE
# constrained-skill-discovery-lab               latest    00b00b647e1b   2 minutes ago       18.9GB
# isaac-lab-base                   latest    892938acb55c   About an hour ago   18.9GB
```

### Running the container

After building, the usual next step is to start the containers associated with your services. You can do this with:

```bash
docker compose --env-file .env.base --file docker-compose.yaml up -d
```

This will start the services defined in your `docker-compose.yaml` file, including constrained-skill-discovery-lab.


### Interacting with a running container

If you want to run commands inside the running container, you can use the `exec` command:

```bash
docker exec --interactive --tty -e DISPLAY=${DISPLAY} constrained-skill-discovery-lab /bin/bash
```

### Shutting down the container

When you are done or want to stop the running containers, you can bring down the services:

```bash
docker compose --env-file .env.base --file docker-compose.yaml down
```

This stops and removes the containers, but keeps the images.

## Training:
To train Ours with Temporal Distance and with normalised states:
```bash
python scripts/skrl_sd/train_hierarchical_Lipschitz.py --headless --max_magnitude 1.5 --sd_version "MSE" --skill_discovery_distance_metric "one" --normalise_encoder_states 1 --param_dual_lambda_lr 5.e-4 --param_dual_lambda_init 1.
```

To train Ours with Euclidean (L2) Distance and with normalised states:

```bash
python scripts/skrl_sd/train_hierarchical_Lipschitz.py --headless --max_magnitude 1. --sd_version "MSE" --skill_discovery_distance_metric "l2" --normalise_encoder_states 1 --param_dual_lambda_lr 1.e-4 --param_dual_lambda_init 30.
```

To train Ours with Euclidean (L2) Distance and with unnormalised states:

```bash
python scripts/skrl_sd/train_hierarchical_Lipschitz.py --headless --max_magnitude 50. --sd_version "MSE" --skill_discovery_distance_metric "l2" --normalise_encoder_states 0 --param_dual_lambda_lr 1.e-4 --param_dual_lambda_init 30.
```

To train base METRA with normalised states:

```bash
python scripts/skrl_sd/train_hierarchical_Lipschitz.py --headless --max_magnitude 1. --sd_version "base" --skill_discovery_distance_metric "one" --normalise_encoder_states 1 --param_dual_lambda_lr 5.e-4 --param_dual_lambda_init 1.
```

To train base LSD with normalised states:

```bash
python scripts/skrl_sd/train_hierarchical_Lipschitz.py --headless --max_magnitude 1. --sd_version "base" --skill_discovery_distance_metric "l2" --normalise_encoder_states 1 --param_dual_lambda_lr 5.e-4 --param_dual_lambda_init 1.
 ```

## Evaluation
To evaluate policies, you can use the provided scripts under `scripts/skrl_sd`. We provide several runs that you can test here:

To evaluate a trained 3D skill policy using temporal distance:
```bash
python scripts/skrl_sd/goal_tracking_task.py --task "Anymal-Skill-Discovery-D-Hierarchical-PLAY" --num_envs 1 --checkpoint "logs/skrl/anymal/3d_temporal_normalised/checkpoints/agent_50000.pt" --max_magnitude 1.5 --skill_discovery_distance_metric "one" --normalise_encoder_states 1 --skill_dim 3
```

To evaluate a trained 3D skill policy using Euclidean distance:
```bash
python scripts/skrl_sd/goal_tracking_task.py --task "Anymal-Skill-Discovery-D-Hierarchical-PLAY" --num_envs 1 --checkpoint "logs/skrl/anymal/3d_l2_unnormalised/checkpoints/agent_50000.pt" --max_magnitude 50. --skill_discovery_distance_metric "l2" --normalise_encoder_states 0 --skill_dim 3
```

To visualise the skills of a trained 2D skill policy using Euclidean (L2) distance:

```bash
python scripts/skrl_sd/run_learned_skills.py --task "Anymal-Skill-Discovery-D-Hierarchical-PLAY" --num_envs 1 --checkpoint "logs/skrl/anymal/2d_l2_normalised/checkpoints/agent_50000.pt" --max_magnitude 1. --skill_discovery_distance_metric "l2" --normalise_encoder_states 1 --skill_dim 2
```

To visualise the skills of a trained 2D skill policy using Temporal distance:

```bash
python scripts/skrl_sd/run_learned_skills.py --task "Anymal-Skill-Discovery-D-Hierarchical-PLAY" --num_envs 1 --checkpoint "logs/skrl/anymal/2d_temporal_normalised/checkpoints/agent_50000.pt" --max_magnitude 1. --skill_discovery_distance_metric "one" --normalise_encoder_states 1 --skill_dim 2
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/exts/constrained_skill_discovery"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
