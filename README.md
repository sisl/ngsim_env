
# NGSIM Env
- This is a rllab environment for learning human driver models with imitation learning
- This repository does not contain a [gail](https://arxiv.org/abs/1606.03476) / [infogail](https://arxiv.org/abs/1703.08840) / hgail implementation
- It also does not contain the human driver data you need for the environment to work. See [NGSIM.jl](https://github.com/sisl/NGSIM.jl) for that.

## Demo
### GAIL in a sing-agent environment
![](single_agent_gail.gif)

### Single agent GAIL and PS-GAIL in a multi-agent environment
![](single_multi_model_2_seed_1.gif)

# Overview
For install directions for ngsim_env and hgail, see the repos, or a guide we adapted and made more robust to issues we encountered: https://github.com/raunakbh92/InstallInstructions/blob/master/install_ngsim_env_hgail.md

### To train and run a single agent GAIL policy: 
0. Navigate to ngsim_env/scripts/imitation
1. Train a policy, this involves running imitate.py (see ngsim_env/docs/training.md) python imitate.py --exp_name NGSIM-gail --n_itr 1000 --policy_recurrent True
2. Validate the policy (this creates trajectories on all NGSIM sections using the trained policy) python validate.py --n_proc 5 --exp_dir ../../data/experiments/NGSIM-gail/ --params_filename itr_1000.npz --random_seed 42
3. Visualize the results:Open up a jupyter notebook and play around with the visualize*.ipynb files. They should be pretty intuitive.

### To reproduce our experiments for the multiagent gail paper submitted to IROS, navigate to ngsim_env/scripts/imitation and read the readme there

## Install
- see [`docs/install.md`](docs/install.md)

## Training
- see [`docs/training.md`](docs/training.md)

## How's this work?
- See README files individual directories for details, but a high-level description is:
- The python code uses [pyjulia](https://github.com/JuliaPy/pyjulia) to instantiate a Julia interpreter, see the `python` directory for details
- The driving environment is then built in Julia, see the `julia` directory for details
- Each time the environment is stepped forward, execution passes from python to julia, updating the environment
