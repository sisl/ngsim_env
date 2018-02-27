using Base.Test
using AutoEnvs

function test_simple_ctor()
    srand(2)
    filepath = Pkg.dir("NGSIM", "data", "trajdata_i80_trajectories-0400-0415.txt")
    params = Dict(
        "trajectory_filepaths"=>[filepath],
        "H"=>40,
        "primesteps"=>5
    )
    env = NGSIMEnv(params)
    x = reset(env)
    nx, r, terminal, _ = step(env, [0.,0.])
    nx, r, terminal, _ = step(env, [0.,0.])
end

function test_basics()
    # ctor
    filepath = Pkg.dir("NGSIM", "data", "trajdata_i80_trajectories-0400-0415.txt")
    params = Dict("trajectory_filepaths"=>[filepath], "H"=>200)
    env = NGSIMEnv(params)

    # reset, step
    x = reset(env)
    a = [0., 0.]
    nx, r, terminal, infos = step(env, a)
    @test x != nx
    @test terminal == false
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
    a = [0., 1.]
    for _ in 1:10
        nx, _, _, _ = step(env, a)
    end
    
    @test abs(nx[acc_idx]) <= 1e-1
    @test abs(nx[tur_idx] - 1) <= 1e-1

    # test infos 
    reset(env)
    _, _, _, infos = step(env, [1.,1.])
    @test infos["rmse_pos"] != 0.

    # test reset with egoid
    x = reset(env; offset=250, egoid=3194, start=8886)
    nx = reset(env; offset=250, egoid=3194, start=8886)
    @test x == nx

end

function test_all_roadways()
    srand(2)
    filenames = [
        "trajdata_i80_trajectories-0400-0415.txt",
        "trajdata_i80_trajectories-0500-0515.txt",
        "trajdata_i80_trajectories-0515-0530.txt",
        "trajdata_i101_trajectories-0805am-0820am.txt",
        "trajdata_i101_trajectories-0820am-0835am.txt",
        "trajdata_i101_trajectories-0750am-0805am.txt"
    ]
    for fn in filenames
        filepath = Pkg.dir("NGSIM", "data", fn)
        params = Dict(
            "trajectory_filepaths"=>[filepath],
            "H"=>50,
            "primesteps"=>5
        )
        env = NGSIMEnv(params)
        x = reset(env)
    end 
end

function test_render()
    srand(2)
    filepath = Pkg.dir("NGSIM", "data", "trajdata_i101_trajectories-0805am-0820am.txt")
    params = Dict(
        "trajectory_filepaths"=>[filepath],
        "H"=>200,
        "primesteps"=>50
    )
    env = NGSIMEnv(params)

    x = reset(env)
    imgs = []
    for _ in 1:100
        a = [1.,0.]
        img = render(env)
        x, r, terminal, _ = step(env, a)
        if terminal
            break
        end
    end
end

# @time test_simple_ctor()
@time test_basics()
# manual tests
# @time test_all_roadways()
# @time test_render() 
