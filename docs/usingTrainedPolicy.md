Assuming you have finished the installation process, and you have access to a trained policy, the following steps can be used
to run a car with the trained policy and visualize the results. 

```bash
cd ~/ngsim_env	# Navigate to your location of ngsim_env. In my case it was in my ~

mkdir data/

mkdir data/experiments

# Now choose a model name. I've used random_model_name as an exemplar here
mkdir data/experiments/random_model_name

mkdir data/experiments/random_model_name/imitate

mkdir data/experiments/random_model_name/imitate/log

cd data/experiments/random_model_name/imitate/log/

# Now download the trained policy files into the current directory (i.e. ngsim_env/data/experiments/random_model_name/imitate/log)
# So now, you have args.npz, iter_200.npz, log.txt as the three files here. These constitute the trained policy

# Now we go for using the trained policy to drive a car
cd ~/ngsim_env/scripts/imitation

# Drive a car using the policy
# Can change n_proc to determine how many cores you want to use
python validate.py --n_proc 1 --exp_dir ../../data/experiments/random_model_name/ --params_filename itr_200.npz --random_seed 42
```
Now, the resulting trajectories have been generated. Next step, visualize the results. 

```bash
# Staying the same directory i.e. ngsim_env/scripts/imitation
jupyter notebook

# The above will open up a jupyter notebook in your browser showing all the files in the current directory. The .ipynb
# files are helpful for visualization as described below
```
- the visualize family of ipynb's have headers at the top of each file describing what it does.
  - visualize.ipynb is for extracting the Root Mean Square Error
  - visualize_trajectories.ipynb creates videos such as the one shown below in the demo section
  - visualize_emergent.ipynb calculates the emergent metrics such as offroad duration and collision rate
