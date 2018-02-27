export 
    Env,
    step,
    reset,
    observation_space_spec,
    action_space_spec,
    render,
    obs_names,
    reward_names,
    vectorized,
    num_envs

abstract type Env end
Base.step(env::Env, action::Int) = error("Not implemented")
Base.step(env::Env, action::Float64) = error("Not implemented")
Base.step(env::Env, action::Array{Float64}) = error("Not implemented")
Base.reset(env::Env) = error("Not implemented")
Base.reset(env::Env; kwargs...) = error("Not implemented")
Base.reset(env::Env, dones::Union{Vector{Bool}, Void}; kwargs...) = reset(env; kwargs...)
observation_space_spec(env::Env) = error("Not implemented")
action_space_spec(env::Env) = error("Not implemented")
render(env::Env) = error("Not implemented")
obs_names(env::Env) = error("Not implemented")
vectorized(env::Env) = false
num_envs(env::Env) = 1