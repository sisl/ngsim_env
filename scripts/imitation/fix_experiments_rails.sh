# First, run the experiments. 3 multiagent models and 3 singleagent models.
# They can all be run in parallel. After they are done, we will generate the validation trajectories.
# There is also the fine tuning step in between.

# RAILS - specify reward augmentation in ngsim_env/julia/AutoEnvs/muliagent_ngsim_env.py, 
#                                        function _extract_rewards()
# REWARD is something like 4000, or could be more involved like col_off_2000_1000
REWARD=3000

start=`date +%s`

EMPTY_ARR=()

# First, CURRICULUM TRAINING
MODELS_TO_FIX_FOR_CURRICULUM=(2)
for num in "${MODELS_TO_FIX_FOR_CURRICULUM[@]}" # policy number
#echo "skipped curriculum"
#for num in "${EMPTY_ARR[@]}" # dont do this loop 
do
    echo "here"
    python multiagent_curriculum_training.py --exp_name multiagent_rails_${REWARD}_${num}_{} \
        --reward_handler_use_env_rewards True &
done

FAIL=0
for job in `jobs -p`
do
	wait $job || let "FAIL+=1"
done

echo "Failed : " $FAIL
end_curr=`date +%s`

# Now, FINE TUNE
MODELS_TO_FIX_FOR_FINETUNE=(1)
for num in "${MODELS_TO_FIX_FOR_FINETUNE[@]}" # policy number
#echo "skipped fine tune"
#for num in "${EMPTY_ARR[@]}" # dont do this loop 
do
    model=multiagent_rails_${REWARD}_$num
    python imitate.py --exp_name ${model}_fine --env_multiagent True \
        --use_infogail False --policy_recurrent True --n_itr 200 --n_envs 100 \
        --validator_render False  --batch_size 40000 --gradient_penalty 2 \
        --discount .99 --recurrent_hidden_dim 64 \
        --params_filepath ../../data/experiments/${model}_50/imitate/log/itr_200.npz \
        --reward_handler_use_env_rewards True &
done

FAIL=0
for job in `jobs -p`
do
	wait $job || let "FAIL+=1"
done

echo "Failed : " $FAIL
end_fine=`date +%s`

# VALIDATE - creates the validation trajectories - simulates the model on each road section
# Does one at a time because already heavily parallelized
#FAIL=0
#for num in "${MODELS_TO_FIX[@]}" # policy number
#echo "skipped validate"
#for num in "${EMPTY_ARR[@]}" # dont do this loop 
#do
#    model=multiagent_rails_${REWARD}_${num}_fine
#    echo $model
#    python validate.py --n_proc 6 --exp_dir ../../data/experiments/${model}/ \
#        --params_filename itr_200.npz --use_multiagent True --random_seed 3 --n_envs 100 &
#
#    for job in `jobs -p`
#    do
#        echo $job
#        wait $job || let "FAIL+=1"
#    done
#done
#echo "Failed : " $FAIL

#Now that validation is done, there will be .npz trajectory files for each of the experiments
# These should appear in ../../data/experiments/{model_name}/imiate/validation/

# Now, you can run visualize.py, or use the visualize ipython notebooks (recommended) to examine the results.

end=`date +%s`

runtime=$((end-start))
runtime_curr=$((end_curr-start))
runtime_fine=$((end_fine-end_curr))
#runtime_validate=$((end-end_fine))

echo "Total, curriculum, fine, validate times: "
echo $runtime
echo $runtime_curr
echo $runtime_fine
#echo $runtime_validate

