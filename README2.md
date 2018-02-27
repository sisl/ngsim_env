A combination of the ngsim_env and hgail repos from wulfebw's github.

To get imports to work correctly while maintaining umbrella project, create symbolic links in home directory:

`cd ~

ln -s [path/to/this/project/]multiagent_gail/hgail/ hgail

ln -s [path/to/this/project/]multiagent_gail/ngsim_env/ ngsim_env `

For example, if you cloned multiagent_gail into home directory (~/), it would simply be:

`cd ~

ln -s ~/multiagent_gail/hgail/ hgail

ln -s ~/multiagent_gail/ngsim_env/ ngsim_env`

For install directions for ngsim_env and hgail, see the repos, or a guide we adapted and made more robust to issues we encountered:
https://github.com/raunakbh92/InstallInstructions/blob/master/install_ngsim_env_hgail.md

To train and run a single agent GAIL policy:
0. Navigate to ngsim_env/scripts/imitation
1. Train a policy, this involves running imitate.py (see ngsim_env/docs/training.md)
    `python imitate.py --exp_name NGSIM-gail --n_itr 1000 --policy_recurrent True`
2. Validate the policy (this creates trajectories on all NGSIM sections using the trained policy)
    `python validate.py --n_proc 5 --exp_dir ../../data/experiments/NGSIM-gail/ --params_filename itr_1000.npz --random_seed 42`
3. Visualize the results
    Open up a jupyter notebook and play around with the visualize*.ipynb files. They should be pretty intuitive.

To reproduce our experiments for the multiagent gail paper submitted to IROS, navigate to ngsim_env/scripts/imitation and read the readme there