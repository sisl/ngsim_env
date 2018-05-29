export
    make

function make(env_id::String, env_params::Dict)
    try
        if env_id == "DeterministicSingleStepDebugEnv"
            return DeterministicSingleStepDebugEnv(env_params)
        elseif env_id == "NGSIMEnv"
            return NGSIMEnv(env_params)
        elseif env_id == "VectorizedNGSIMEnv"
            return VectorizedNGSIMEnv(env_params)
        elseif env_id == "MultiagentNGSIMEnv"
            return MultiagentNGSIMEnv(env_params)
	elseif env_id == "MultiagentNGSIMEnvVideoMaker"
	    	#println("RAUNAK make.jl says video maker")
		return MultiagentNGSIMEnvVideoMaker(env_params)
        else
            throw(ArgumentError("Invalid env_id: $(env_id)"))
        end
    catch e
        println("exception raised while making environment")
        println(backtrace(e))
        println(e)
        rethrow(e)
    end
end
