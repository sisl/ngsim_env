
import numpy as np
import os
import unittest

from julia_env.julia_env import JuliaEnv

class TestNGSIMEnv(unittest.TestCase):

    def test_ngsim_env(self):
        basedir = os.path.expanduser('~/.julia/packages/NGSIM/9OYUa/data')
        filename = 'trajdata_i80_trajectories-0400-0415.txt'
        filepaths = [os.path.join(basedir, filename)]
        env = JuliaEnv(
            env_id='NGSIMEnv',
            env_params=dict(trajectory_filepaths=filepaths),
            using='AutoEnvs'
        )
        x = env.reset()
        nx, r, t, info = env.step(np.array([0.,0.]))
        self.assertTrue(np.sum(np.abs(x-nx)) > 1e-1)

        # complex reset
        x = env.reset(offset=250, egoid=3194, start=8886)
        nx = env.reset(offset=250, egoid=3194, start=8886)
        np.testing.assert_array_almost_equal(x, nx, 4)

if __name__ == '__main__':
    unittest.main()