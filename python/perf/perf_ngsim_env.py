
import numpy as np
import os

from julia_env.julia_env import JuliaEnv

from context_timer import ContextTimer

def perf_ngsim_env_step():
    basedir = os.path.expanduser('~/.julia/v0.6/NGSIM/data')
    filename = 'trajdata_i101_trajectories-0750am-0805am.txt'
    filepaths = [os.path.join(basedir, filename)]
    env = JuliaEnv(
        env_id='NGSIMEnv',
        env_params=dict(
            trajectory_filepaths=filepaths,
        ),
        using='AutoEnvs'
    )
    n_steps = 10000
    action = np.array([1.,0.])
    env.reset()
    with ContextTimer():
        for _ in range(n_steps):
            _, _, terminal, _ = env.step(action)
            if terminal:
                env.reset()

def perf_vectorized_ngsim_env_step():
    basedir = os.path.expanduser('~/.julia/v0.6/NGSIM/data')
    filename = 'trajdata_i101_trajectories-0750am-0805am.txt'
    filepaths = [os.path.join(basedir, filename)]
    n_envs = 100
    env = JuliaEnv(
        env_id='VectorizedNGSIMEnv',
        env_params=dict(
            n_envs=n_envs,
            trajectory_filepaths=filepaths,
        ),
        using='AutoEnvs'
    )
    n_steps = 10000
    action = np.zeros((n_envs, 2))
    action[:,0] = 1.
    env.reset()
    with ContextTimer():
        for _ in range(0, n_steps, n_envs):
            _, _, terminal, _ = env.step(action)
            env.reset(terminal)

if __name__ == '__main__':
    perf_ngsim_env_step()
    perf_vectorized_ngsim_env_step()