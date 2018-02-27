export 
    DebugEnvDeterministicSingleStep,
    reset,
    step,
    observation_space_spec,
    action_space_spec

#=
Description:
    Takes a single step from initial state 0. 
    Actions are 1 or 2.
    Action 1 yields 1 reward.
    Action 2 yields 2 reward.
    Terminates after the single action.
=#
type DeterministicSingleStepDebugEnv <: Env
    function DeterministicSingleStepDebugEnv(params::Dict=Dict())
        return new()
    end
end
Base.reset(env::DeterministicSingleStepDebugEnv) = 0
function Base.step(env::DeterministicSingleStepDebugEnv, action::Int)
    @assert action == 1 || action == 2
    return (0, action, true, Dict())
end
observation_space_spec(env::DeterministicSingleStepDebugEnv) = (1,), "Discrete"
action_space_spec(env::DeterministicSingleStepDebugEnv) = (2,), "Discrete"
render(env::DeterministicSingleStepDebugEnv) = nothing

