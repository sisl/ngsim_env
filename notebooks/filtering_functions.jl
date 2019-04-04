"""
compute_particle_likelihoods: Loop over the particles and score each of them

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

------------Arguments that need explanation:
`p_set_dict` Dictionary with parameters of IDM as keys and associated value as array of particles
`f` Frame to start hallucination from
`trupos` Resulting true position starting from frame f
`approach` Select "pf" or "cem"
`elite_fraction_percent` Required for the cem method to fit a distribution

------------Other functions called:`compute_particle_likelihoods`

------------Returns:
`new_p_set_dict` Dictionary with keys as IDM parameters and values as array of particles

-----------NOTES
- This function updates associated particles over 1 step for one car
- I think frame and scene can be used as the same thing. Maybe techincally scene is an array
with each element in that array being a frame.
- This function will be called by a function that loops over all the cars present in a scene
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
