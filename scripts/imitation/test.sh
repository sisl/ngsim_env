FAIL = 0
for i in 1 25 50 75 100;
do
    echo $i

    for model in 2_fine; #run 2 at a time
    do
        echo $model
        echo python validate.py --n_proc 20 --exp_dir ../../data/experiments/multiagent_curr_$model/ --params_filename itr_200.npz --use_multiagent True --random_seed 3 --n_envs $i &

    done
    
done

echo $FAIL
