# Documentation
- This directory contains scripts for extracting NGSIM trajectories for use as the expert data and for running imitation learning

## NGSIM Expert Data Extraction
```julia
julia extract_ngsim_demonstrations.jl
```

## Imitation
- The main script for running imitation learning is `imitation/imitate.py`
- This script requires the gail implementation provided in the `hgail` python package
- See `imitation/hyperparams.py` for the training options and default hyperparameters
- To train a policy in the multiagent environment, run
```bash
python imitate.py --env_multiagent True --use_infogail False --exp_name NGSIM-multiagent --n_itr 1000 --policy_recurrent True
```
- This command
  + `--env_multiagent True` Runs imitation in the multiagent environment
  + `--use_infogail False` Does not use infogail (i.e., uses gail)
  + `--exp_name NGSIM-multiagent` Stores results in a directory named NGSIM-multiagent (this directory should be created in `ngsim_env/data/experiments/`)
  + `--n_itr 1000` Runs 1000 TRPO iterations
  + `--policy_recurrent True` Uses a recurrent policy
