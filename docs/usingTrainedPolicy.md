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
