
import julia
import os

from rllab.envs.base import Env

from julia_env.utils import build_space

class JuliaEnv(Env):

    def __init__(self, env_id, env_params, using):

        # initialize julia
        self.j = julia.Julia()
        self.j.eval('include(\"{}\")'.format(os.path.expanduser('~/.juliarc.jl')))
        self.j.using(using)

        # initialize environment
        self.env = self.j.make(env_id, env_params)
        self._observation_space = build_space(*self.j.observation_space_spec(self.env))
        self._action_space = build_space(*self.j.action_space_spec(self.env))

    def reset(self, dones=None, **kwargs):
        return self.j.reset(self.env, dones, **kwargs)

    def step(self, action):
        return self.j.step(self.env, action)

    def render(self, *args, **kwargs):
        return self.j.render(self.env, *args, **kwargs)
        
    def obs_names(self):
        return self.j.obs_names(self.env)

    def vec_env_executor(self, *args, **kwargs):
        return self

    @property
    def num_envs(self):
        return self.j.num_envs(self.env)

    @property 
    def vectorized(self):
        return self.j.vectorized(self.env)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space
