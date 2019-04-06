"""
compute_particle_likelihoods: Loop over the particles and score each of them

------Idea------------
Compute the likelihood of the true position under a gaussian distribution
centered at the hallucinated position and with standard deviation given
according to the particle sigma parameter

-----Other functions called: `to_matrix_form`,`hallucinate_a_step`
"""
function compute_particle_likelihoods(roadway,f,trupos,p_set_dict;car_id=-1)
    if car_id==-1 @show "Please give valid car_id" end
    timestep = 0.1 #TODO: Remove hardcoding
    p_mat, params, vec_val_vec = to_matrix_form(p_set_dict)
    
    num_params=size(p_mat)[1]
    num_p = size(p_mat)[2]
    lkhd_vec = Array{Float64}(undef,num_p)
    for i in 1:num_p    
        # Create dict version for a single particle
        p_dict = Dict()
        for j in 1:num_params
            p_dict[params[j]]=vec_val_vec[j][i]
        end
        
        std_dev_acc = p_dict[:σ]
        
        # hack to avoid the std_dev_pos become negative and error Normal distb
        if std_dev_acc <= 0 std_dev_acc = 0.1 end
        
        # TODO: This math needs to be verified from random variable calculations
        std_dev_pos = timestep*timestep*std_dev_acc
            
        hpos = hallucinate_a_step(roadway,f,p_dict,car_id=car_id)
        lkhd_vec[i] = pdf(Normal(hpos,std_dev_pos),trupos[1])
    end
    return lkhd_vec,p_mat,params
end

"""
update_p_one_step: Update particles given one step of true data
- This function updates associated particles over 1 step for one car
- This function will be called by a function that loops over all the cars present in a scene

--------------Idea flow---------------
PF:
Compute particle likelihoods->assign weights to particles, higher likelihood higher the weight
->resample particles according the the weight

CEM:
Compute particle likelihoods->sort particles by highest to lowest likelihood->
select elites->fit a new distribution using these elites->sample new particles

------------Arguments that need explanation:
`p_set_dict` Dictionary with parameters of IDM as keys and associated value as array of particles
`f` Frame to start hallucination from
`trupos` Resulting true position starting from frame f
`approach` Select "pf" or "cem"
`elite_fraction_percent` Required for the cem method to fit a distribution

------------Other functions called:`compute_particle_likelihoods`

------------Returns:
`new_p_set_dict` Dictionary with keys as IDM parameters and values as array of particles
"""
function update_p_one_step(roadway,f,trupos,p_set_dict;
                            car_id=-1,approach="pf",elite_fraction_percent=20)
    if car_id==-1 @show "Provide valid car_id" end
    
    lkhd_vec,p_mat,params = compute_particle_likelihoods(roadway,f,trupos,p_set_dict,car_id=car_id)
    
    num_params = size(p_mat)[1]
    num_p = size(p_mat)[2]
    
    if approach=="pf"
        p_weight_vec = weights(lkhd_vec./sum(lkhd_vec)) # Convert to weights form to use julia sampling
        idx = sample(1:num_p,p_weight_vec,num_p)
        new_p_mat = p_mat[:,idx] #Careful that idx is (size,1) and not (size,2)
    end
    
    if approach=="cem"
        sortedidx = sortperm(lkhd_vec,rev=true)
        numtop = convert(Int64,ceil(num_p*elite_fraction_percent/100.0))
        best_particles = p_mat[:,sortedidx[1:numtop]] # elite selection
#         @show best_particles
        p_distribution = fit(MvNormal,best_particles) # fit distb using elites
        new_p_mat = rand(p_distribution,num_p) # sample num_p new particles from dist
    end
    
    new_p_set_dict = to_dict_form(params,new_p_mat)
    return new_p_set_dict
end

"""
    filter_particles_over_trajectory(num_particles::Int64,num_cars::Int64,lane_place_array::Array,
        car_particles::Array,particle_props::Array;approach="pf")

Runs particle filtering over an entire trajectory of provided ground truth data.
Places cars on road, generates truth traj using provided true parameter values, generates bucket with candidate
particles for each car, runs particle filtering over the entire trajectory and outputs estimated parameters

# Arguments
- `lane_place_array::Array`: Every element corresponds to a new lane starting
from lane1. Every element is an array. The number of elements in this
array is the number of cars in the same lane. Every element is a tuple.
Each tuple contains pos,vel for the car
- `car_particles::Array`: Array with each element being a dictionary with true particle corresponding
to car_id equivalent to that index
- `particle_props`: Array with each element corresponding to a different parameter.
Each element is a tuple with 4 elements. These are
symbol with param name, start value to sample from, step, end value to sample
- `approach = "pf"`: The filtering approach to be used. Allows "pf" and "cem"
"""
function filter_particles_over_trajectory(num_particles::Int64,num_cars::Int64,lane_place_array::Array,
        car_particles::Array,particle_props::Array;approach="pf")
    scene,roadway = init_place_cars(lane_place_array)
    rec = generate_truth_data(lane_place_array,car_particles)
    f_end_num = length(rec.frames)
    
    bucket_array = initialize_carwise_particle_buckets(n_cars,num_p,particle_props)
    for t in 1:f_end_num-1
        if t%10==0 @show t end
        f = rec.frames[f_end_num - t + 1]

        for car_id in 1:n_cars
            old_p_set_dict = bucket_array[car_id]
            trupos = rec.frames[f_end_num-t].entities[car_id].state.posF.s
            new_p_set_dict = update_p_one_step(roadway,f,trupos,old_p_set_dict,
                car_id=car_id,approach=approach)
            bucket_array[car_id] = new_p_set_dict
        end
    end  
    #@show fit(MvNormal,old_p_mat) # Don't work because all elements identical
    print_buckets_mean(bucket_array)
    # @show bucket_array
end
