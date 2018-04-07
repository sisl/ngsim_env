# Sometimes fine tune runs fail, so this is run to fix those. 


python imitate.py --env_multiagent True --use_infogail False --exp_name multiagent_rails_col_off_2000_1_50 --n_itr 200 --policy_recurrent True --n_envs 50 --params_filepath ../../data/experiments/multiagent_rails_col_off_2000_1_40/imitate/log/itr_200.npz --validator_render False --reward_handler_use_env_rewards True

FAIL=0
for job in `jobs -p`
do
	wait $job || let "FAIL+=1"
done

echo $FAIL

# Now, FINE TUNE
for num in 1; 
do
    model=${num}_fine
    python imitate.py --exp_name multiagent_rails_col_off_2000_$model --env_multiagent True \
        --use_infogail False --policy_recurrent True --n_itr 200 --n_envs 100 \
        --validator_render False  --batch_size 40000 --gradient_penalty 2 \
        --discount .99 --recurrent_hidden_dim 64 \
        --params_filepath ../../data/experiments/multiagent_rails_col_off_2000_${num}_50/imitate/log/itr_200.npz \
        --reward_handler_use_env_rewards True &
done

FAIL=0
for job in `jobs -p`
do
	wait $job || let "FAIL+=1"
done

echo $FAIL

# VALIDATE - creates the validation trajectories - simulates the model on each road section
FAIL=0
for model in 1_fine; 
do
    python validate.py --n_proc 20 --exp_dir ../../data/experiments/multiagent_rails_col_off_2000_$model/ \
        --params_filename itr_200.npz --use_multiagent True --random_seed 3 --n_envs 100 &

    for job in `jobs -p`
    do
        echo $job
        wait $job || let "FAIL+=1"
    done
done

#Now that validation is done, there will be .npz trajectory files for each of the experiments
# These should appear in ../../data/experiments/{model_name}/imiate/validation/

# Now, you can run visualize.py, or use the visualize ipython notebooks (recommended) to examine the results.

