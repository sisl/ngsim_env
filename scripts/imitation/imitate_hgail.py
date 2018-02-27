
import numpy as np
import os
import tensorflow as tf

from hgail.algos.hgail_impl import HGAIL

import auto_validator
import hyperparams
import utils

# setup
args = hyperparams.parse_args()
exp_dir = utils.set_up_experiment(exp_name=args.exp_name, phase='imitate')
saver_dir = os.path.join(exp_dir, 'imitate', 'log')
saver_filepath = os.path.join(saver_dir, 'checkpoint')
np.savez(os.path.join(saver_dir, 'args'), args=args)
summary_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'imitate', 'summaries'))

# build components
env, act_low, act_high = utils.build_ngsim_env(args, exp_dir, vectorize=args.vectorize)
data = utils.load_data(
    args.expert_filepath, 
    act_low=act_low, 
    act_high=act_high, 
    min_length=args.env_H + args.env_primesteps,
    clip_std_multiple=args.normalize_clip_std_multiple,
    ngsim_filename=args.ngsim_filename
)
critic = utils.build_critic(args, data, env, summary_writer)
hierarchy = utils.build_hierarchy(args, env, summary_writer)
validator = auto_validator.AutoValidator(
    summary_writer, 
    data['obs_mean'], 
    data['obs_std'],
    flat_recurrent=args.policy_recurrent
)
saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=.5)
algo = HGAIL(
    critic=critic, 
    hierarchy=hierarchy,
    saver=saver,
    saver_filepath=saver_filepath,
    validator=validator
)

# session for actual training
with tf.Session() as session:
 
    # running the initialization here to allow for later loading
    # NOTE: rllab batchpolopt runs this before training as well 
    # this means that any loading subsequent to this is nullified 
    # you have to comment of that initialization for any loading to work
    session.run(tf.global_variables_initializer())

    # loading
    if args.params_filepath != '':
        algo.load(args.params_filepath)

    # run training
    algo.train(sess=session)
