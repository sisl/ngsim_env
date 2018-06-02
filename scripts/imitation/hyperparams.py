'''
default hyperparameters for training
these are build as args to allow for command line options 
these args are also saved along with parameters during training to 
allow for rebuilding everything with the same settings
'''

import argparse
import numpy as np

from utils import str2bool

def parse_args(arglist=None):
    parser = argparse.ArgumentParser()
   
    # decaying reward logistics
    parser.add_argument('--decay_reward', type=str2bool, default=False)
    parser.add_argument('--exp_dir', type=str, default='../../data/experiments')
    parser.add_argument('--itrs_per_decay', type=int, default=25)

    # curriculum params
    parser.add_argument('--do_curriculum', type=str2bool, default=False)
    parser.add_argument('--n_envs_start', type=int, default=10)
    parser.add_argument('--n_envs_end', type=int, default=50)
    parser.add_argument('--n_envs_step', type=int, default=10)
    parser.add_argument('--load_params_init', type=str, default='NONE') 
    # if not the string 'NONE', inserted into first parampath for curriculum

    # logistics
    parser.add_argument('--exp_name', type=str, default='NGSIM-gail')
    parser.add_argument('--params_filepath', type=str, default='')
    parser.add_argument('--expert_filepath', type=str, default='../../data/trajectories/ngsim.h5')
    parser.add_argument('--vectorize', type=str2bool, default=True)
    parser.add_argument('--n_envs', type=int, default=50)
    parser.add_argument('--normalize_clip_std_multiple', type=float, default=10.)

    # env
    parser.add_argument('--ngsim_filename', type=str, default='trajdata_i101_trajectories-0750am-0805am.txt')
    parser.add_argument('--env_H', type=int, default=200)
    parser.add_argument('--env_primesteps', type=int, default=50)
    parser.add_argument('--env_action_repeat', type=int, default=1)
    parser.add_argument('--env_multiagent', type=str2bool, default=False)
    parser.add_argument('--env_reward', type=int, default=0)

    # reward handler
    parser.add_argument('--reward_handler_max_epochs', type=int, default=100)
    parser.add_argument('--reward_handler_recognition_final_scale', type=float, default=.2)
    parser.add_argument('--reward_handler_use_env_rewards', type=str2bool, default=True)
    parser.add_argument('--reward_handler_critic_final_scale', type=float, default=1.)

    # policy 
    parser.add_argument('--use_infogail', type=str2bool, default=True)
    parser.add_argument('--policy_mean_hidden_layer_dims', nargs='+', default=(128,128,64))
    parser.add_argument('--policy_std_hidden_layer_dims', nargs='+', default=(128,64))
    parser.add_argument('--policy_recurrent', type=str2bool, default=False)
    parser.add_argument('--recurrent_hidden_dim', type=int, default=64)

    # critic
    parser.add_argument('--use_critic_replay_memory', type=str2bool, default=True)
    parser.add_argument('--n_critic_train_epochs', type=int, default=40)
    parser.add_argument('--critic_learning_rate', type=float, default=.0004)
    parser.add_argument('--critic_dropout_keep_prob', type=float, default=.8)
    parser.add_argument('--gradient_penalty', type=float, default=2.)
    parser.add_argument('--critic_grad_rescale', type=float, default=40.)
    parser.add_argument('--critic_batch_size', type=int, default=1000)
    parser.add_argument('--critic_hidden_layer_dims', nargs='+', default=(128,128,64))

    # recognition
    parser.add_argument('--latent_dim', type=int, default=4)
    parser.add_argument('--n_recognition_train_epochs', type=int, default=30)
    parser.add_argument('--scheduler_k', type=int, default=20)
    parser.add_argument('--recognition_learning_rate', type=float, default=.0005)
    parser.add_argument('--recognition_hidden_layer_dims', nargs='+', default=(128,64))

    # gail
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--trpo_step_size', type=float, default=.01)
    parser.add_argument('--n_itr', type=int, default=2000)
    parser.add_argument('--max_path_length', type=int, default=1000)
    parser.add_argument('--discount', type=float, default=.95)

    # render
    parser.add_argument('--validator_render', type=str2bool, default=False)
    parser.add_argument('--render_every', type=int, default=25)
    parser.add_argument('--remove_ngsim_veh', type=str2bool, default=False)


    # parse and return
    if arglist is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(arglist)
    return args

def load_args(args_filepath):
    '''
    This function enables backward-compatible usage of saved args files by 
    filling in missing values with default values.
    '''
    orig = np.load(args_filepath)['args'].item()
    new = parse_args(arglist=[])
    orig_keys = set(orig.__dict__.keys())
    new_keys = list(new.__dict__.keys())
    # replace all keys in both orig and new, in new, with orig values
    for k in new_keys:
        if k in orig_keys:
            new.__dict__[k] = orig.__dict__[k]
    return new
