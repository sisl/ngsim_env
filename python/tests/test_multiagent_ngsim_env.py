
import numpy as np
import os
import tensorflow as tf
import unittest

from rllab.baselines.zero_baseline import ZeroBaseline

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

from julia_env.julia_env import JuliaEnv

class TestMultiagentNGSIMEnv(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.set_random_seed(1)
        np.random.seed(1)
        tf.reset_default_graph() 

    def test_multiagent_ngsim_env(self):
        basedir = os.path.expanduser('~/.julia/v0.6/NGSIM/data')
        filename = 'trajdata_i101_trajectories-0750am-0805am.txt'
        filepaths = [os.path.join(basedir, filename)]
        n_veh = 5
        env = JuliaEnv(
            env_id='MultiagentNGSIMEnv',
            env_params=dict(
                n_veh=n_veh, 
                trajectory_filepaths=filepaths,
                H=200,
                primesteps=50
            ),
            using='AutoEnvs'
        )
        low, high = env.action_space.low, env.action_space.high
        env = TfEnv(env)
        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_sizes=(32,32),
            std_hidden_sizes=(32,32),
            adaptive_std=True,
            output_nonlinearity=None,
            learn_std=True
        )
        baseline = ZeroBaseline(env_spec=env.spec)
        algo = TRPO(
            env=env, 
            policy=policy, 
            baseline=baseline,
            n_itr=1,
            batch_size=1000,
            sampler_args=dict(
                n_envs=n_veh
            )
        )
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                algo.train(sess=sess)
            except Exception as e:
                self.fail('exception incorrectly raised: {}'.format(e))

if __name__ == '__main__':
    unittest.main()