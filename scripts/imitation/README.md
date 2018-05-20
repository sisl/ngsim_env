# Imitation scripts

This directory is where the things really happen. The control center if you will. 

Below is a (very) short way you can reproduce our experiments, and below that is a breakdown of the important files.

Assuming everything is set up correctly (a strong assumption):

$ ./run_experiments.sh

Yeah that's really it. I set that up to run our experiments, but all of the hyperparameters are not necessarily specified at all times, some remain default, so if the results are drastically different there could be a few sources of error:

* Different defaults in hyperparams.py
* Someone changed multiagent_curriculum_training.py
* Some other file was changed, possibly from the gail algorithm
* Randomness, although this should not drastically affect the results, since we did run a number of models. The overall performance should be consistent, but we did see high variance for the single agent for example.

It should take on the order of 1-2 days to run everything, and will take up a decent amount of space (mostly the validation trajectories, about 12GB for each model if validating on all 6 NGSIM time periods). The models will be saved to ngsim_env/data/experiments, so make sure there is space available there.

After running that, go through the visualize ipynb's and create the results.

### Important Files

run_experiments.sh
 * runs the experiments we ran, same hyperparameters and experiment names
 
imitate.py
 * where the tensorflow session is established for training, and all of that jazz.
 
validate.py
 * simulates a given policy on specified environments. 
 
hyperparams.py
 * defines the default hyperparameters for use elsewhere.
 
visualize*.ipynb
 * the visualize family of ipynb's have headers at the top of each file describing what it does.
 * visualize is for rmse
 * visualize_trajectories creates videos
 * visualize_emergent calculates the emergent metrics.
 
multiagent_curriculum_training.py
 * this is where we implement curriculum training for the multiagent models. It is pretty straightforward, but essentially it runs a model by initially training with 10 agents for 200 iterations, then using that policy as the initalization for the next model trained on 20 agents, and so forth. 
 * You can specify the things like n_itrs each, n_envs_start, and n_envs_end, but we used the default.

run_n_agents_vs_perf_tests.sh
 * This bash script runs a combination of validate.py (doing a multiagent and singleagent in parallel, 20 processes each) and visualize.py, generating rmse_[attr]_[nagents].png files (in this directory). 


### Troubleshooting
#### Not necessarily a fully fleshed out guide at the moment, more of just keeping track of issues we encounter.
Running validate.py occasionally hangs with no error messages or anything like that. 
Previous experience suggests that this is somehow related to julia processes remaining unfinished and the python
script moving on. Looking in validate.py, there is a sleep() call. In the past we have had some limited success in 
overcoming the hanging problem by increasing the sleep duration. However, it is not guaranteed.
We have been unable to produce a minimum reproducible example of this happening, but the thoughts are that it is
related to the machine's load. A higher load means we need to wait longer.
