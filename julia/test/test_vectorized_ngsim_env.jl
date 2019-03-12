using Test
using AutoEnvs
using NGSIM

function test_basics()
    # ctor
    filepath = joinpath(dirname(pathof(NGSIM)), "..", "data", "trajdata_i80_trajectories-0400-0415.txt")
    n_envs = 100
    params = Dict("trajectory_filepaths"=>[filepath], "n_envs"=>n_envs)
    env = VectorizedNGSIMEnv(params)

    # reset, step
    x = reset(env)
    a = zeros(n_envs, 2)
    nx, r, terminal, infos = step(env, a)
    @test x != nx
    nnx, r, terminal, infos = step(env, a)
    @test nx != nnx

    # obs spec
    shape, spacetype, infos = observation_space_spec(env)
    @test spacetype == "Box"
    @test in("high", keys(infos))
    @test in("low", keys(infos))

    for _ in 1:200
        _, _, terminals, _ = step(env, a)
        reset(env, terminals)
    end
    
end

#@time test_basics()
