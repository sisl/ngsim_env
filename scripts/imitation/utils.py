
import h5py
import numpy as np
import os
import tensorflow as tf

from rllab.envs.base import EnvSpec
from rllab.envs.normalized_env import normalize as normalize_env
import rllab.misc.logger as logger

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.spaces.discrete import Discrete

from hgail.algos.hgail_impl import Level
from hgail.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from hgail.critic.critic import WassersteinCritic
from hgail.envs.spec_wrapper_env import SpecWrapperEnv
from hgail.envs.vectorized_normalized_env import vectorized_normalized_env
from hgail.misc.datasets import CriticDataset, RecognitionDataset
from hgail.policies.categorical_latent_sampler import CategoricalLatentSampler
from hgail.policies.gaussian_latent_var_gru_policy import GaussianLatentVarGRUPolicy
from hgail.policies.gaussian_latent_var_mlp_policy import GaussianLatentVarMLPPolicy
from hgail.policies.latent_sampler import UniformlyRandomLatentSampler
from hgail.core.models import ObservationActionMLP
from hgail.policies.scheduling import ConstantIntervalScheduler
from hgail.recognition.recognition_model import RecognitionModel
from hgail.samplers.hierarchy_sampler import HierarchySampler
import hgail.misc.utils

from julia_env.julia_env import JuliaEnv

'''
Const
'''
NGSIM_FILENAME_TO_ID = {
    'trajdata_i101_trajectories-0750am-0805am.txt': 1,
    'trajdata_i101_trajectories-0805am-0820am.txt': 2,
    'trajdata_i101_trajectories-0820am-0835am.txt': 3,
    'trajdata_i80_trajectories-0400-0415.txt': 4,
    'trajdata_i80_trajectories-0500-0515.txt': 5,
    'trajdata_i80_trajectories-0515-0530.txt': 6
}

'''
Common 
'''
def maybe_mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

def partition_list(lst, n):
    sublists = [[] for _ in range(n)]
    for i, v in enumerate(lst):
        sublists[i % n].append(v)
    return sublists

def str2bool(v):
    if v.lower() == 'true':
        return True
    return False

def write_trajectories(filepath, trajs):
    np.savez(filepath, trajs=trajs)

def load_trajectories(filepath):
    return np.load(filepath)['trajs']

def filename2label(fn):
    s = fn.find('-') + 1
    e = fn.rfind('_')
    return fn[s:e]

def load_trajs_labels(directory, files_to_use=[0,1,2,3,4,5]):
    filenames = [
        'trajdata_i101_trajectories-0750am-0805am_trajectories.npz',
        'trajdata_i101_trajectories-0805am-0820am_trajectories.npz',
        'trajdata_i101_trajectories-0820am-0835am_trajectories.npz',
        'trajdata_i80_trajectories-0400-0415_trajectories.npz',
        'trajdata_i80_trajectories-0500-0515_trajectories.npz',
        'trajdata_i80_trajectories-0515-0530_trajectories.npz'
    ]
    filenames = [filenames[i] for i in files_to_use]
    labels = [filename2label(fn) for fn in filenames]
    filepaths = [os.path.join(directory, fn) for fn in filenames]
    trajs = [load_trajectories(fp) for fp in filepaths]
    return trajs, labels

'''
Component build functions
'''

'''
This is about as hacky as it gets, but I want to avoid editing the rllab 
source code as much as possible, so it will have to do for now.

Add a reset(self, kwargs**) function to the normalizing environment
https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
'''
def normalize_env_reset_with_kwargs(self, **kwargs):
    ret = self._wrapped_env.reset(**kwargs)
    if self._normalize_obs:
        return self._apply_normalize_obs(ret)
    else:
        return ret

def add_kwargs_to_reset(env):
    normalize_env = hgail.misc.utils.extract_normalizing_env(env)
    if normalize_env is not None:
        normalize_env.reset = normalize_env_reset_with_kwargs.__get__(normalize_env)

'''end of hack, back to our regularly scheduled programming'''

# Raunak adding an input argument for multiagent video making
def build_ngsim_env(
        args,
        exp_dir='/tmp', 
        alpha=0.001,
        vectorize=False,
        render_params=None,
        videoMaking=False):
    basedir = os.path.expanduser('~/.julia/packages/NGSIM/9OYUa/data')
    filepaths = [os.path.join(basedir, args.ngsim_filename)]
    if render_params is None:
        render_params = dict(
            viz_dir=os.path.join(exp_dir, 'imitate/viz'),
            zoom=5.
        )
    env_params = dict(
        trajectory_filepaths=filepaths,
        H=args.env_H,
        primesteps=args.env_primesteps,
        action_repeat=args.env_action_repeat,
        terminate_on_collision=False,
        terminate_on_off_road=False,
        render_params=render_params,
        n_envs=args.n_envs,
        n_veh=args.n_envs,
        remove_ngsim_veh=args.remove_ngsim_veh,
        reward=args.env_reward
    )
    # order matters here because multiagent is a subset of vectorized
    # i.e., if you want to run with multiagent = true, then vectorize must 
    # also be true

    if args.env_multiagent:
        env_id = 'MultiagentNGSIMEnv'
        if videoMaking:
            print('RAUNAK BHATTACHARRYA VIDEO MAKER IS ON')
            env_id='MultiagentNGSIMEnvVideoMaker'
        alpha = alpha * args.n_envs
        normalize_wrapper = vectorized_normalized_env
    elif vectorize:
        env_id = 'VectorizedNGSIMEnv'
        alpha = alpha * args.n_envs
        normalize_wrapper = vectorized_normalized_env

    else:
        env_id = 'NGSIMEnv'
        normalize_wrapper = normalize_env

    env = JuliaEnv(
        env_id=env_id,
        env_params=env_params,
        using='AutoEnvs'
    )
    # get low and high values for normalizing _real_ actions
    low, high = env.action_space.low, env.action_space.high
    env = TfEnv(normalize_wrapper(env, normalize_obs=True, obs_alpha=alpha))
    add_kwargs_to_reset(env)
    return env, low, high

def build_critic(args, data, env, writer=None):
    if args.use_critic_replay_memory:
        critic_replay_memory = hgail.misc.utils.KeyValueReplayMemory(maxsize=3 * args.batch_size)
    else:
        critic_replay_memory = None

    critic_dataset = CriticDataset(
        data, 
        replay_memory=critic_replay_memory,
        batch_size=args.critic_batch_size,
        flat_recurrent=args.policy_recurrent
    )

    critic_network = ObservationActionMLP(
        name='critic', 
        hidden_layer_dims=args.critic_hidden_layer_dims,
        dropout_keep_prob=args.critic_dropout_keep_prob
    )
    critic = WassersteinCritic(
        obs_dim=env.observation_space.flat_dim,
        act_dim=env.action_space.flat_dim,
        dataset=critic_dataset, 
        network=critic_network,
        gradient_penalty=args.gradient_penalty,
        optimizer=tf.train.RMSPropOptimizer(args.critic_learning_rate),
        n_train_epochs=args.n_critic_train_epochs,
        summary_writer=writer,
        grad_norm_rescale=args.critic_grad_rescale,
        verbose=2,
        debug_nan=True
    )
    return critic

def build_policy(args, env, latent_sampler=None):
    if args.use_infogail:
        if latent_sampler is None:
            latent_sampler = UniformlyRandomLatentSampler(
                scheduler=ConstantIntervalScheduler(k=args.scheduler_k),
                name='latent_sampler',
                dim=args.latent_dim
            )
        if args.policy_recurrent:
            policy = GaussianLatentVarGRUPolicy(
                name="policy",
                latent_sampler=latent_sampler,
                env_spec=env.spec,
                hidden_dim=args.recurrent_hidden_dim,
            )
        else:
            policy = GaussianLatentVarMLPPolicy(
                name="policy",
                latent_sampler=latent_sampler,
                env_spec=env.spec,
                hidden_sizes=args.policy_mean_hidden_layer_dims,
                std_hidden_sizes=args.policy_std_hidden_layer_dims
            )
    else:
        if args.policy_recurrent:
            policy = GaussianGRUPolicy(
                name="policy",
                env_spec=env.spec,
                hidden_dim=args.recurrent_hidden_dim,
                output_nonlinearity=None,
                learn_std=True
            )
        else:
            policy = GaussianMLPPolicy(
                name="policy",
                env_spec=env.spec,
                hidden_sizes=args.policy_mean_hidden_layer_dims,
                std_hidden_sizes=args.policy_std_hidden_layer_dims,
                adaptive_std=True,
                output_nonlinearity=None,
                learn_std=True
            )
    return policy

def build_recognition_model(args, env, writer=None):
    if args.use_infogail:
        recognition_dataset = RecognitionDataset(
            args.batch_size,
            flat_recurrent=args.policy_recurrent
        )
        recognition_network = ObservationActionMLP(
            name='recog', 
            hidden_layer_dims=args.recognition_hidden_layer_dims,
            output_dim=args.latent_dim
        )
        recognition_model = RecognitionModel(
            obs_dim=env.observation_space.flat_dim,
            act_dim=env.action_space.flat_dim,
            dataset=recognition_dataset, 
            network=recognition_network,
            variable_type='categorical',
            latent_dim=args.latent_dim,
            optimizer=tf.train.AdamOptimizer(args.recognition_learning_rate),
            n_train_epochs=args.n_recognition_train_epochs,
            summary_writer=writer,
            verbose=2
        )
    else:
        recognition_model = None
    return recognition_model

def build_baseline(args, env):
    return GaussianMLPBaseline(env_spec=env.spec)

def build_reward_handler(args, writer=None):
    reward_handler = hgail.misc.utils.RewardHandler(
        use_env_rewards=args.reward_handler_use_env_rewards,
        max_epochs=args.reward_handler_max_epochs, # epoch at which final scales are used
        critic_final_scale=args.reward_handler_critic_final_scale,
        recognition_initial_scale=0.,
        recognition_final_scale=args.reward_handler_recognition_final_scale,
        summary_writer=writer,
        normalize_rewards=True,
        critic_clip_low=-100,
        critic_clip_high=100,
    )
    return reward_handler

def build_hierarchy(args, env, writer=None):
    levels = []

    latent_sampler = UniformlyRandomLatentSampler(
        name='base_latent_sampler',
        dim=args.latent_dim,
        scheduler=ConstantIntervalScheduler(k=args.env_H)
    )
    for level_idx in [1,0]:
        # wrap env in different spec depending on level
        if level_idx == 0:
            level_env = env
        else:
            level_env = SpecWrapperEnv(
                env,
                action_space=Discrete(args.latent_dim),
                observation_space=env.observation_space
            )
            
        with tf.variable_scope('level_{}'.format(level_idx)):
            # recognition_model = build_recognition_model(args, level_env, writer)
            recognition_model = None
            if level_idx == 0:
                policy = build_policy(args, env, latent_sampler=latent_sampler)
            else:
                scheduler = ConstantIntervalScheduler(k=args.scheduler_k)
                policy = latent_sampler = CategoricalLatentSampler(
                    scheduler=scheduler,
                    name='latent_sampler',
                    policy_name='latent_sampler_policy',
                    dim=args.latent_dim,
                    env_spec=level_env.spec,
                    latent_sampler=latent_sampler,
                    max_n_envs=args.n_envs
                )
            baseline = build_baseline(args, level_env)
            if args.vectorize:
                force_batch_sampler = False
                if level_idx == 0:
                    sampler_args = dict(n_envs=args.n_envs)
                else:
                    sampler_args = None
            else:
                force_batch_sampler = True
                sampler_args = None

            sampler_cls = None if level_idx == 0 else HierarchySampler
            algo = TRPO(
                env=level_env,
                policy=policy,
                baseline=baseline,
                batch_size=args.batch_size,
                max_path_length=args.max_path_length,
                n_itr=args.n_itr,
                discount=args.discount,
                step_size=args.trpo_step_size,
                sampler_cls=sampler_cls,
                force_batch_sampler=force_batch_sampler,
                sampler_args=sampler_args,
                optimizer_args=dict(
                    max_backtracks=50,
                    debug_nan=True
                )
            )
            reward_handler = build_reward_handler(args, writer)
            level = Level(
                depth=level_idx,
                algo=algo,
                reward_handler=reward_handler,
                recognition_model=recognition_model,
                start_itr=0,
                end_itr=0 if level_idx == 0 else np.inf
            )
            levels.append(level)

    # by convention the order of the levels should be increasing
    # but they must be built in the reverse order 
    # so reverse the list before returning it
    return list(reversed(levels))

'''
setup
'''

def latest_snapshot(exp_dir, phase='train'):
    snapshot_dir = os.path.join(exp_dir, phase, 'log')
    snapshots = glob.glob('{}/itr_*.pkl'.format(snapshot_dir))
    latest = sorted(snapshots, reverse=True)[0]
    return latest

def set_up_experiment(
        exp_name, 
        phase, 
        exp_home='../../data/experiments/',
        snapshot_gap=5):
    maybe_mkdir(exp_home)
    exp_dir = os.path.join(exp_home, exp_name)
    maybe_mkdir(exp_dir)
    phase_dir = os.path.join(exp_dir, phase)
    maybe_mkdir(phase_dir)
    log_dir = os.path.join(phase_dir, 'log')
    maybe_mkdir(log_dir)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode('gap')
    logger.set_snapshot_gap(snapshot_gap)
    log_filepath = os.path.join(log_dir, 'log.txt')
    logger.add_text_output(log_filepath)
    return exp_dir

'''
data utilities
'''

def compute_lengths(arr):
    sums = np.sum(np.array(arr), axis=2)
    lengths = []
    for sample in sums:
        zero_idxs = np.where(sample == 0.)[0]
        if len(zero_idxs) == 0:
            lengths.append(len(sample))
        else:
            lengths.append(zero_idxs[0])
    return np.array(lengths)

def normalize(x, clip_std_multiple=np.inf):
    mean = np.mean(x, axis=0, keepdims=True)
    x = x - mean
    std = np.std(x, axis=0, keepdims=True) + 1e-8
    up = std * clip_std_multiple
    lb = - std * clip_std_multiple
    x = np.clip(x, lb, up)
    x = x / std
    return x, mean, std

def normalize_range(x, low, high):
    low = np.array(low)
    high = np.array(high)
    mean = (high + low) / 2.
    half_range = (high - low) / 2.
    x = (x - mean) / half_range
    x = np.clip(x, -1, 1)
    return x

def load_x_feature_names(filepath, ngsim_filename):
    f = h5py.File(filepath, 'r')
    xs = []
    traj_id = NGSIM_FILENAME_TO_ID[ngsim_filename]
    # in case this nees to allow for multiple files in the future
    traj_ids = [traj_id]
    for i in traj_ids:
        if str(i) in f.keys():
            xs.append(f[str(i)])
        else:
            raise ValueError('invalid key to trajectory data: {}'.format(i))
    x = np.concatenate(xs)
    feature_names = f.attrs['feature_names']
    return x, feature_names

def load_data(
        filepath,
        act_keys=['accel', 'turn_rate_global'],
        ngsim_filename='trajdata_i101_trajectories-0750am-0805am.txt',
        debug_size=None,
        min_length=50,
        normalize_data=True,
        shuffle=False,
        act_low=-1,
        act_high=1,
        clip_std_multiple=np.inf):
    
    # loading varies based on dataset type
    x, feature_names = load_x_feature_names(filepath, ngsim_filename)

    # optionally keep it to a reasonable size
    if debug_size is not None:
        x = x[:debug_size]
       
    if shuffle:
        idxs = np.random.permutation(len(x))
        x = x[idxs]

    # compute lengths of the samples before anything else b/c this is fragile
    lengths = compute_lengths(x)

    # flatten the dataset to (n_samples, n_features)
    # taking only the valid timesteps from each sample
    # i.e., throw out timeseries information
    xs = []
    for i, l in enumerate(lengths):
        # enforce minimum length constraint
        if l >= min_length:
            xs.append(x[i,:l])
    x = np.concatenate(xs)

    # split into observations and actions
    # redundant because the environment is not able to extract actions
    obs = x
    act_idxs = [i for (i,n) in enumerate(feature_names) if n in act_keys]
    act = x[:, act_idxs]

    if normalize_data:

        # normalize it all, _no_ test / val split
        obs, obs_mean, obs_std = normalize(obs, clip_std_multiple)
        # normalize actions to between -1 and 1
        act = normalize_range(act, act_low, act_high)

    else:
        obs_mean = None
        obs_std = None

    return dict(
        observations=obs,
        actions=act,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
