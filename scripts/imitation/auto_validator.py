
import io
import numpy as np
import tensorflow as tf

import sys
import matplotlib
backend = 'Agg' if sys.platform == 'linux' else 'TkAgg'
matplotlib.use(backend)
import matplotlib.pyplot as plt

from rllab.envs.normalized_env import NormalizedEnv
from rllab.sampler.utils import rollout

from hgail.envs.vectorized_normalized_env import VectorizedNormalizedEnv
from hgail.misc.validator import Validator
from hgail.misc.rollout import vectorized_render_rollout
import hgail.misc.utils

from julia_env.julia_env import JuliaEnv

def plt2imgsum():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_sum = tf.Summary.Image(encoded_image_string=buf.getvalue())
    plt.clf()
    return img_sum

class AutoValidator(Validator):

    def __init__(
            self, 
            writer, 
            obs_mean, 
            obs_std,
            render=True,
            render_every=25,
            flat_recurrent=False):
        super(AutoValidator, self).__init__(writer)
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.render = render
        self.render_every = render_every
        self.flat_recurrent = flat_recurrent

    def _summarize_env_infos(self, env_infos):
        summaries = []

        # means 
        mean_keys = ['rmse_pos', 'rmse_vel', 'rmse_t', 'is_colliding']
        for key in mean_keys:
            mean = np.mean(env_infos[key])
            tag = 'validation/mean_{}'.format(key)
            summaries += [tf.Summary.Value(tag=tag, simple_value=mean)]

        # hist
        hist_keys = ['rmse_pos', 'rmse_vel', 'rmse_t']
        for key in hist_keys:
            plt.hist(np.reshape(env_infos[key], -1), 50)
            img_sum = plt2imgsum()
            tag = 'validation/hist_{}'.format(key)
            summaries += [tf.Summary.Value(tag=tag, image=img_sum)]

        return summaries

    def _summarize_actions(self, actions):
        summaries = []

        _, act_dim = actions.shape
        for i in range(act_dim):
            plt.hist(actions[:,i], 50)
            img_sum = plt2imgsum()
            tag = 'validation/hist_action_{}'.format(i)
            summaries += [tf.Summary.Value(tag=tag, image=img_sum)]

        return summaries

    def _summarize_latent(self, samples_data):
        summaries = []
        latent = samples_data['agent_infos']['latent']
        actions = hgail.misc.utils.flatten(samples_data['actions'])
        if len(latent.shape) == 3:
            latent = np.reshape(latent, (-1, latent.shape[-1]))
        n_samples, latent_dim = latent.shape
        action_dim = actions.shape[1]
        # histogram actions, distringuishing based on latent value
        # assumes discrete latent space
        for l in range(latent_dim):
            idxs = np.where(latent[:,l] == 1.)[0]
            cur_actions = actions[idxs]
            for a in range(action_dim):
                plt.hist(cur_actions[:,a], 50)
                img_sum = plt2imgsum()
                tag = 'validation/hist_action_{}_latent_{}'.format(a, l)
                summaries += [tf.Summary.Value(tag=tag, image=img_sum)]
                tag = 'validation/mean_action_{}_latent_{}'.format(a, l)
                mean = np.mean(cur_actions[:,a])
                summaries += [tf.Summary.Value(tag=tag, simple_value=mean)]
        return summaries

    def _summarize_samples_data(self, samples_data):
        summaries = []
        if 'env_infos' in samples_data.keys():
            summaries += self._summarize_env_infos(samples_data['env_infos'])
        if self.flat_recurrent:
            actions = hgail.misc.utils.flatten(samples_data['actions'])
        else:
            actions = samples_data['actions']
        summaries += self._summarize_actions(actions)

        if 'agent_infos' in samples_data.keys() and 'latent' in samples_data['agent_infos'].keys():
            summaries += self._summarize_latent(samples_data)

        return summaries

    def _summarize_obs_mean_std(self, env_mean, env_std, true_mean, true_std, labels):
        summaries = []
        mean_diff = np.reshape(env_mean, -1) - np.reshape(true_mean, -1)
        std_diff = np.reshape(env_std, -1) - np.reshape(true_std, -1)
        for i, label in enumerate(labels):
            tag = 'comparison/mean_diff_{}'.format(label)
            summaries += [tf.Summary.Value(tag=tag, simple_value=mean_diff[i])]
            tag = 'comparison/std_diff_{}'.format(label)
            summaries += [tf.Summary.Value(tag=tag, simple_value=std_diff[i])]

        tag = 'comparison/overall_abs_mean_diff'
        summaries += [tf.Summary.Value(tag=tag, simple_value=np.mean(np.abs(mean_diff)))]
        tag = 'comparison/overall_abs_std_diff'
        summaries += [tf.Summary.Value(tag=tag, simple_value=np.mean(np.abs(std_diff)))]

        return summaries

    def validate(self, itr, objs):
        summaries = []
        keys = objs.keys()

        if 'samples_data' in keys:
            summaries += self._summarize_samples_data(objs['samples_data'])

        if 'env' in keys:
            # extract some relevant, wrapped environments
            normalized_env = hgail.misc.utils.extract_wrapped_env(objs['env'], NormalizedEnv)
            if normalized_env is None:
                normalized_env = hgail.misc.utils.extract_wrapped_env(objs['env'], VectorizedNormalizedEnv)
            julia_env = hgail.misc.utils.extract_wrapped_env(objs['env'], JuliaEnv)

            summaries += self._summarize_obs_mean_std(
                normalized_env._obs_mean, 
                np.sqrt(normalized_env._obs_var),
                self.obs_mean,
                self.obs_std,
                julia_env.obs_names()
            )

        # render a trajectory, this must save to file on its own
        if self.render and 'env' in keys and 'policy' in keys and (itr % self.render_every) == 0:
            if objs['env'].vectorized:
                vectorized_render_rollout(objs['env'], objs['policy'], max_path_length=200)
            else:
                rollout(objs['env'], objs['policy'], animated=True, max_path_length=200)

        self.write_summaries(itr, summaries)
        
