# Raunak makes a script to only run validation given fine trained policy

BASE_NAME="continuous_normalized_laneid"

# RAILS - specify reward augmentation in ngsim_env/julia/AutoEnvs/muliagent_ngsim_env.py, 
#                                        function _extract_rewards()
# REWARD is something like 4000, or could be more involved like col_off_2000_1000
REWARD=0
# TODO don't forget to change it in the file!!

# VALIDATE - creates the validation trajectories - simulates the model on each road section
for num in 1; 
do
    model=${BASE_NAME}_${REWARD}_${num}_fine
    python validate.py --n_proc 7 --exp_dir ../../data/experiments/${model}/ \
        --params_filename itr_200.npz --use_multiagent True --random_seed 3 --n_envs 100 

done

for num in 2; 
do
    model=${BASE_NAME}_${REWARD}_${num}_fine
    python validate.py --n_proc 7 --exp_dir ../../data/experiments/${model}/ \
        --params_filename itr_200.npz --use_multiagent True --random_seed 3 --n_envs 100 

done

for num in 3; 
do
    model=${BASE_NAME}_${REWARD}_${num}_fine
    python validate.py --n_proc 7 --exp_dir ../../data/experiments/${model}/ \
        --params_filename itr_200.npz --use_multiagent True --random_seed 3 --n_envs 100 

done


#Now that validation is done, there will be .npz trajectory files for each of the experiments
# These should appear in ../../data/experiments/{model_name}/imiate/validation/
