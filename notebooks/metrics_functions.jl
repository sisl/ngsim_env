"""
    calc_rmse_pos(truerec::QueueRecord,simrec::QueueRecord;num_cars=-1)

Calculate the RMSE in position between two QueueRecords i.e. what you get after running
simulate. Use case is that truerec is the ground truth trajectory and the other is the
trajectory you get after running with the estimated IDM parameters

# Returns
- `rmse_pos::Array`: An array with the rmse_pos indexed by time
"""
function calc_rmse_pos(truerec,simrec;num_cars=-1)
    @assert num_cars != -1
    n_frames = length(truerec.frames)
    @assert length(truerec.frames) == length(simrec.frames)
    
    n_steps = length(truerec.frames)

    X = Array{Float64}(undef,n_steps, 1)
    rmse_pos = Array{Float64}(undef,n_steps,1)
    for t in 1:n_steps
        truef = truerec.frames[n_steps - t + 1]
        simf = simrec.frames[n_steps - t + 1]

        temp_square_error = 0
        for c in 1:num_cars
            trues = truef.entities[c].state.posF.s
            sims = simf.entities[c].state.posF.s

            temp_square_error += sqrt(abs2(trues-sims))
    #         @show temp_square_error
        end
        rmse_pos[t] = temp_square_error/num_cars
    end
    return rmse_pos
end

"""
    particle_difference(trueparticle::Dict,particle::Dict)
Find Euclidean distance between two dictionaries having same keys

# Returns
norm of the vector created by the difference between corresponding keys of the
two dictionaries i.e. same parameters of true particle and our candidate particle
"""
function particle_difference(trueparticle::Dict,particle::Dict)
    @assert keys(trueparticle)==keys(particle)
    
    # Create a dictinary with same keys as input dicts but value as diff between
    diff_particle = merge(-,trueparticle,particle)
    
    # Find the norm of the vector containing these diff values
    return norm(collect(values(diff_particle)))
end

"""
    particle_difference(trueparticle::Dict,particle::Dict)
Find norm difference between each key of two dicts with same keys

# Returns
- `diff_particle::Dict` Dict with keys same as input dicts and values as abs(diff) between two for every key
"""
function particle_difference_paramwise(trueparticle::Dict,particle::Dict)
    @assert keys(trueparticle)==keys(particle)
    
    # Create a dictinary with same keys as input dicts but value as diff between
    diff_particle = merge(-,trueparticle,particle)
    
    # Find the norm of the vector containing these diff values
    for (k,v) in diff_particle
        diff_particle[k] = norm(v)
    end
    return diff_particle
end
