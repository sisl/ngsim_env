export 
    VectorizedNGSIMEnv,
    reset,
    step,
    observation_space_spec,
    action_space_spec,
    obs_names,
    render,
    vectorized,
    num_envs

#=
Description:
    NGSIM env that plays NGSIM trajectories, allowing the agent to take the place 
    of one of the vehicles in the trajectory
=#
type VectorizedNGSIMEnv <: Env
    n_envs::Int
    envs::Vector{NGSIMEnv}
    x::Array{Float64}
    r::Vector{Float64}
    dones::Vector{Bool}
    
    function VectorizedNGSIMEnv(
            params::Dict; 
            n_envs::Int = 10,
            reclength::Int = 10,
            Δt::Float64 = .1,
            primesteps::Int = 50,
            H::Int = 50,
            terminate_on_collision::Bool = true,
            terminate_on_off_road::Bool = true,
            render_params::Dict = Dict("zoom"=>5., "viz_dir"=>"/tmp"))
        param_keys = keys(params)
        @assert in("trajectory_filepaths", param_keys)

        # optionally overwrite defaults
        n_envs = get(params, "n_envs", n_envs)
        reclength = get(params, "reclength", reclength)
        primesteps = get(params, "primesteps", primesteps)
        H = get(params, "H", H)
        for (k,v) in get(params, "render_params", render_params)
            render_params[k] = v
        end
        terminate_on_collision = get(params, "terminate_on_collision", terminate_on_collision)
        terminate_on_off_road = get(params, "terminate_on_off_road", terminate_on_off_road)

        # load trajdatas
        trajdatas, trajinfos, roadways = load_ngsim_trajdatas(
            params["trajectory_filepaths"],
            minlength=primesteps + H
        )

        # build envs
        envs = Vector{NGSIMEnv}(n_envs)
        for i in 1:n_envs
            envs[i] = NGSIMEnv(
                params,
                trajdatas=trajdatas, 
                trajinfos=trajinfos,
                roadways=roadways,
                reclength=reclength,
                Δt=Δt,
                primesteps=primesteps,
                H=H,
                terminate_on_collision=terminate_on_collision,
                terminate_on_off_road=terminate_on_off_road,
                render_params=render_params
            )
        end
        n_features = length(obs_names(envs[1]))
        x = zeros(n_envs, n_features)
        r = zeros(n_envs)
        dones = Vector{Bool}(n_envs)

        return new(n_envs, envs, x, r, dones)
    end
end
function reset(env::VectorizedNGSIMEnv, dones::Vector{Bool} = fill!(Vector{Bool}(env.n_envs), true))
    for i in 1:env.n_envs
        if dones[i]
            env.x[i, :] = reset(env.envs[i])
        end
    end
    return deepcopy(env.x)
end 
function Base.step(env::VectorizedNGSIMEnv, actions::Array{Float64})
    infos = Vector{Dict}(env.n_envs)
    for i in 1:env.n_envs
            env.x[i, :], env.r[i], env.dones[i], infos[i] = step(env.envs[i], actions[i, :])
    end
    # vectorized sampler does not call reset on the environment
    # but expects the environment to handle resetting, so do that here
    # note: this mutates env.x in order to return the correct obs when resetting
    reset(env, env.dones)
    return deepcopy(env.x), env.r, env.dones, stack_tensor_dict_list(infos)
end

observation_space_spec(env::VectorizedNGSIMEnv) = observation_space_spec(env.envs[1])
action_space_spec(env::VectorizedNGSIMEnv) = action_space_spec(env.envs[1])
obs_names(env::VectorizedNGSIMEnv) = obs_names(env.envs[1])
render(env::VectorizedNGSIMEnv) = render(env.envs[1])
vectorized(env::VectorizedNGSIMEnv) = true
num_envs(env::VectorizedNGSIMEnv) = env.n_envs