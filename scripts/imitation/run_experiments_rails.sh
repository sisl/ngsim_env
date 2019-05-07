# First, run the experiments. 3 multiagent models and 3 singleagent models.
# They can all be run in parallel. After they are done, we will generate the validation trajectories.
# There is also the fine tuning step in between.

# RAILS - specify reward augmentation in ngsim_env/julia/src/muliagent_ngsim_env.py, 
#                                        function _extract_rewards()
# REWARD is something like 4000, or could be more involved like col_off_2000_1000
REWARD=2000

BASE_NAME="rails_smoothed"
# smoothing of the rewards is accomplished by smoothing the features. 
# See julia/src/multiagent_ngsm_inv.py, function _compute_feature_infos()

NUM_ITRS=200
DECAY=False
ITRS_PER_DECAY=200
if [ "$DECAY" = True ]; then
    BASE_NAME="${ITRS_PER_DECAY}_decay_${BASE_NAME}"
    echo $BASE_NAME
else
    ITRS_PER_DECAY="$NUM_ITRS" # To be safe, if you don't want to decay, have this == num_itrs
fi

INFOGAIL=True
if [ "$INFOGAIL" = True ]; then
    BASE_NAME="${BASE_NAME}_infogail"
    echo $BASE_NAME
fi

LOG_FILE="logs/${BASE_NAME}_${REWARD}.log"

start=`date +%s`

MODELS_FOR_CURRICULUM=('1' '2' '3')
PARAMS_FOR_CURRICULUM=('NONE', 'NONE', 'NONE')                # default is 'NONE', last completed training

N_ENVS_PER=10                               # shouldn't need to change this
END_CURRICULUM=50                           # End curriculum training with this many envs
                                            # these are useful when fails partway through curriculum
MODELS_FOR_FINETUNE=('1' '2' '3')
MODELS_FOR_VALIDATE=('1' '2' '3')

# First, CURRICULUM TRAiINING
i=0
for num in "${MODELS_FOR_CURRICULUM[@]}" # policy number
do
    python imitate.py --do_curriculum True --exp_name ${BASE_NAME}_${REWARD}_${num} \
        --n_envs_start $((PARAMS_FOR_CURRICULUM[i]+$N_ENVS_PER)) \
        --n_envs_end $END_CURRICULUM --n_itr $NUM_ITRS \
        --policy_recurrent True --gradient_penalty 2 --discount 0.95 \
        --env_multiagent True --use_infogail $INFOGAIL \
        --env_reward $REWARD --decay_reward $DECAY --itrs_per_decay $ITRS_PER_DECAY \
        --load_params_init ${PARAMS_FOR_CURRICULUM[i]} &
    echo "Curriculum policy # ${num}, job id $!, time $(`echo date`)" >> $LOG_FILE
    let "i++"
done

FAIL=0
for job in `jobs -p`
do
	wait $job || let "FAIL+=1"
    echo "Curriculum job id: $job, failed: $FAIL" >> $LOG_FILE
done

echo "Curriculum - Failed : " $FAIL, time: $(`echo date`) >> $LOG_FILE
end_curr=`date +%s`

# Now, FINE TUNE
if [ "$DECAY" = True ]; then
    LAST_DECAY_ITR=$((NUM_ITRS-ITRS_PER_DECAY)) # cumulative number of iterations done on last decay chunk.
    PARAMPATH_END="${END_CURRICULUM}_${LAST_DECAY_ITR}/imitate/log/itr_${ITRS_PER_DECAY}.npz"
else
    PARAMPATH_END="${END_CURRICULUM}/imitate/log/itr_${NUM_ITRS}.npz"
fi

for num in "${MODELS_FOR_FINETUNE[@]}"; 
do
    model=${BASE_NAME}_${REWARD}_$num
    python imitate.py --exp_name ${model}_fine --env_multiagent True \
        --use_infogail $INFOGAIL --policy_recurrent True \
        --n_itr $NUM_ITRS --n_envs 100 \
        --validator_render False  --batch_size 40000 --gradient_penalty 2 \
        --discount .99 --recurrent_hidden_dim 64 \
        --params_filepath ../../data/experiments/${model}_${PARAMPATH_END} \
        --env_reward $REWARD --decay_reward $DECAY --itrs_per_decay $ITRS_PER_DECAY &
    echo "Fine tune policy # ${num}, job id $!, time $(`echo date`)" >> $LOG_FILE
done

FAIL=0
for job in `jobs -p`
do
	wait $job || let "FAIL+=1"
    echo "Fine tune job id: $job, failed: $FAIL" >> $LOG_FILE
done

echo "Fine Tune - Failed : " $FAIL, time: $(`echo date`) >> $LOG_FILE
end_fine=`date +%s`

# VALIDATE - creates the validation trajectories - simulates the model on each road section
FAIL=0
for num in "${MODELS_FOR_VALIDATE[@]}"; 
do
    model="${BASE_NAME}_${REWARD}_${num}_fine"
    if [ "$DECAY" = True ]; then
        model="${model}_${LAST_DECAY_ITR}"
    fi
    python validate.py --n_proc 15 --exp_dir ../../data/experiments/${model}/ \
        --params_filename itr_${ITRS_PER_DECAY}.npz --use_multiagent True \
        --random_seed 3 --n_envs 100 &

    echo "Validate policy # ${num}, job id $!, time $(`echo date`)" >> $LOG_FILE
    for job in `jobs -p`
    do
        wait $job || let "FAIL+=1"
        echo "Validate job id: $job, failed: $FAIL", time: $(`echo date`) >> $LOG_FILE
    done
done
echo "Validate - Failed : " $FAIL, time: $(`echo date`) >> $LOG_FILE

#Now that validation is done, there will be .npz trajectory files for each of the experiments
# These should appear in ../../data/experiments/{model_name}/imiate/validation/

# Now, you can run visualize.py, or use the visualize ipython notebooks (recommended) to examine the results.

end=`date +%s`

runtime=$((end-start))
runtime_curr=$((end_curr-start))
runtime_fine=$((end_fine-end_curr))
runtime_validate=$((end-end_fine))

echo "Total, curriculum, fine, validate times: " >> $LOG_FILE
echo $runtime >> $LOG_FILE
echo $runtime_curr >> $LOG_FILE
echo $runtime_fine >> $LOG_FILE
echo $runtime_validate >> $LOG_FILE

