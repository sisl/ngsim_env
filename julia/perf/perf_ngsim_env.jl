using AutoEnvs

function perf_ngsim_env_step(n_steps=20000)
    filepath = Pkg.dir("NGSIM", "data", "trajdata_i101_trajectories-0750am-0805am.txt")
    params = Dict(
        "trajectory_filepaths"=>[filepath],
    )
    env = NGSIMEnv(params)
    action = [1.,0.]
    reset(env)
    @time for _ in 1:n_steps
        _, _, terminal, _ = step(env, action)
        if terminal
            reset(env)
        end
    end
end

function perf_vectorized_ngsim_env_step(n_steps=20000, n_envs=10)
    filepath = Pkg.dir("NGSIM", "data", "trajdata_i101_trajectories-0750am-0805am.txt")
    params = Dict(
        "trajectory_filepaths"=>[filepath],
    )
    env = VectorizedNGSIMEnv(n_envs, params)
    actions = zeros(2, n_envs)
    actions[1,:] = 1.
    reset(env)
    @time for _ in 1:n_envs:n_steps
        _, _, terminals, _ = AutoEnvs.step(env, actions)
        reset(env, terminals)
    end
end

perf_ngsim_env_step(20000)
perf_vectorized_ngsim_env_step(20000, 100)
