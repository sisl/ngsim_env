using Base.Test
using AutoEnvs

function test_basics()
    # ctor
    filepath = Pkg.dir("NGSIM", "data", "trajdata_i80_trajectories-0400-0415.txt")
    n_veh = 2
    params = Dict(
        "trajectory_filepaths"=>[filepath], 
        "H"=>50,
        "n_veh"=>n_veh
    )
    env = MultiagentNGSIMEnv(params)

    # reset, step
    x = reset(env)
    @test size(x) == (2, 66)
    a = zeros(n_veh, 2)
    nx, r, terminal, infos = step(env, a)
    @test x != nx
    @test terminal == [false, false]
    nnx, r, terminal, infos = step(env, a)
    @test nx != nnx

    # obs spec
    shape, spacetype, infos = observation_space_spec(env)
    @test spacetype == "Box"
    @test in("high", keys(infos))
    @test in("low", keys(infos))
    
    # does accel reflect applied value
    fns = obs_names(env)
    acc_idx = [i for (i,n) in enumerate(fns) if "accel" == n][1]
    tur_idx = [i for (i,n) in enumerate(fns) if "turn_rate_global" == n][1]
    a = zeros(n_veh, 2)
    a[:, 2] = 1
    for _ in 1:10
        nx, _, _, _ = step(env, a)
    end

    @test all(abs.(nx[:, acc_idx]) .<= 1e-1)
    @test all(abs.(nx[:, tur_idx] - 1) .<= 1e-1)

    # test infos 
    reset(env)
    _, _, _, infos = step(env, ones(n_veh, 2))
    @test all(infos["rmse_pos"] .!= 0.)

    # test multiple episodes
    reset(env, [true for _ in 1:n_veh])
    for _ in 1:50
        nx, _, terminal, _ = step(env, a)
    end
    @test all(terminal)
    _, _, terminal, _ = step(env, a)
    @test all(.!terminal)
    for _ in 1:49
        _, _, terminal, _ = step(env, a)
    end
    @test all(terminal)

end

function test_render()
    srand(2)
    n_veh = 50
    filepath = Pkg.dir("NGSIM", "data", "trajdata_i101_trajectories-0805am-0820am.txt")
    params = Dict(
        "trajectory_filepaths"=>[filepath],
        "H"=>50,
        "primesteps"=>50,
        "n_veh"=>n_veh,
        "remove_ngsim_veh"=>true
    )
    env = MultiagentNGSIMEnv(params)

    reset(env)
    a = ones(n_veh, 2)
    for _ in 1:50
        render(env)
        x, r, terminal, _ = step(env, a)
    end
end

@time test_basics()
# manual tests
# @time test_render() 
