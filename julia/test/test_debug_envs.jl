

function test_debug_envs()
    env = AutoEnvs.DeterministicSingleStepDebugEnv()
    x = reset(env)
    nx, r, t, _ = step(env, 1)
end

@time test_debug_envs()