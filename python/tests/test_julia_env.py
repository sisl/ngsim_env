
import unittest

from julia_env.julia_env import JuliaEnv

class TestJuliaEnv(unittest.TestCase):

    def test_julia_env(self):
        env = JuliaEnv(
            env_id='DeterministicSingleStepDebugEnv',
            env_params={},
            using='AutoEnvs'
        )
        x = env.reset()
        nx, r, t, info = env.step(1)

        self.assertEqual(x, 0)
        self.assertEqual(nx, 0)
        self.assertEqual(r, 1)
        self.assertEqual(t, True)

if __name__ == '__main__':
    unittest.main()