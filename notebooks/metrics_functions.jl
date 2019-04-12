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

"""
    estimate_then_evaluate_imitation(num_p::Int64,n_cars::Int64,lane_place_array::Array,
        car_particles::Array,particle_props::Array;approach="pf")

Estimate parameters of IDM, then use the mean of the particle bucket for each car to generate
simulated trajectory and calculate the rmse position between true and simulated trajectory

# Returns
- `rmse_pos_array::Array` Array with each element being the rmse position at that time index
"""
function estimate_then_evaluate_imitation(num_p::Int64,n_cars::Int64,lane_place_array::Array,
        car_particles::Array,particle_props::Array;approach="pf")
    scene,roadway = init_place_cars(lane_place_array)
    rec = generate_truth_data(lane_place_array,car_particles)
    f_end_num = length(rec.frames)
    
    bucket_array = initialize_carwise_particle_buckets(n_cars,num_p,particle_props)
    for t in 1:f_end_num-1
        #if t%10==0 @show t end
        f = rec.frames[f_end_num - t + 1]

        for car_id in 1:n_cars
            old_p_set_dict = bucket_array[car_id]
            trupos = rec.frames[f_end_num-t].entities[car_id].state.posF.s
            new_p_set_dict = update_p_one_step(roadway,f,trupos,old_p_set_dict,
                car_id=car_id,approach=approach)
            bucket_array[car_id] = new_p_set_dict
        end
    end  
    
    # Estimation of parameters ends here. Now generate simulation
    car_particles_sim = find_mean_particle_carwise(bucket_array)
    sim_rec = generate_truth_data(lane_place_array,car_particles_sim)
    
    # Now find the rmse error between the positions of the trajs
    rmse_pos_array = calc_rmse_pos(rec,sim_rec,num_cars = n_cars)
    
    return rmse_pos_array
end
