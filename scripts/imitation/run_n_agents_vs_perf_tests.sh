FAIL = 0
for i in 1 25 50 75 100;
do
    echo $i

    for model in 1_fine 2_fine 3_fine; #run 2 at a time
    do
        echo $model
        python validate.py --n_proc 20 --exp_dir ../../data/experiments/multiagent_curr_$model/ --params_filename itr_200.npz --use_multiagent True --random_seed 3 --n_envs $i &

        python validate.py --n_proc 20 --exp_dir ../../data/experiments/singleagent_def_$model/ --params_filename itr_200.npz --use_multiagent True --random_seed 3  --n_envs $i &

        for job in `jobs -p`
        do
            echo $job
            wait $job || let "FAIL+=1"
        done
    done
    
    python visualize.py $i
done

echo $FAIL
