using Test
using Distributions
using AutomotiveDrivingModels
using LinearAlgebra
using StatsBase # For weights function used to create weighted likelihood
using AutoViz # For SceneOverlay within visualization_functions.jl
using Interact # For @manipulate within visualization_functions.jl
using Reel
using Plots

# Bring in the method definitions
include("admin_functions.jl")
include("driving_functions.jl")
include("filtering_functions.jl")
include("metrics_functions.jl")
include("visualization_functions.jl")

function trueparam_dict2array(a::Dict{Symbol,Float64})
	truparam = []	
	for (k,v) in a
		push!(truparam,v)
	end
	return truparam
end

function viz_particle_dist(num_p::Int64,n_cars::Int64,lane_place_array::Array,
        car_particles::Array,particle_props::Array)
    scene,roadway = init_place_cars(lane_place_array)
    rec = generate_truth_data(lane_place_array,car_particles)
    f_end_num = length(rec.frames)
    
    # -------------vanilla particle filter--------------
    plots_pf = []
    bucket_array = initialize_carwise_particle_buckets(n_cars,num_p,particle_props)
    for t in 1:f_end_num-1
        #if t%10==0 @show t end
        f = rec.frames[f_end_num - t + 1]

        for car_id in 1:n_cars
            old_p_set_dict = bucket_array[car_id]
            trupos = rec.frames[f_end_num-t].entities[car_id].state.posF.s
            new_p_set_dict = update_p_one_step(roadway,f,trupos,old_p_set_dict,
                car_id=car_id,approach="pf")
            bucket_array[car_id] = new_p_set_dict

		# Plot distribution of particles	
		if car_id == 1
			true_param_dict = car_particles[car_id]
			trueparams = trueparam_dict2array(true_param_dict)
			particles = to_particleMatrix(new_p_set_dict)
			push!(plots_pf,plot_particles(particles,trueparams))
		end
        end
    end  
    
    # Estimation of parameters ends here. Now generate simulation
    car_particles_pf = find_mean_particle_carwise(bucket_array)
    sim_rec_pf = generate_truth_data(lane_place_array,car_particles_pf)
    
    # -------------CEM particle filter--------------
    bucket_array = initialize_carwise_particle_buckets(n_cars,num_p,particle_props)
    for t in 1:f_end_num-1
        #if t%10==0 @show t end
        f = rec.frames[f_end_num - t + 1]

        for car_id in 1:n_cars
            old_p_set_dict = bucket_array[car_id]
            trupos = rec.frames[f_end_num-t].entities[car_id].state.posF.s
            new_p_set_dict = update_p_one_step(roadway,f,trupos,old_p_set_dict,
                car_id=car_id,approach="cem")
            bucket_array[car_id] = new_p_set_dict
        end
    end  
    
    # Estimation of parameters ends here. Now generate simulation
    car_particles_cem = find_mean_particle_carwise(bucket_array)
    sim_rec_cem = generate_truth_data(lane_place_array,car_particles_cem)
    
    return rec,sim_rec_pf,sim_rec_cem,roadway, plots_pf
end # End of viz function defined


# Run experiments
# Scenario 2: 5 cars in adjacent lanes
"""
	num_particles = 100
	lane_place_array = [[(0.,10.)],[(0.,20.)],[(0.,15.)],[(0.,20.)],[(0.,20.)]]
	num_cars = 5
	d1 = Dict(:v_des=>10.0,:σ=>0.2);d2 = Dict(:v_des=>20.0,:σ=>0.3);d3 = Dict(:v_des=>15.0,:σ=>0.)
	d4 = Dict(:v_des=>18.0,:σ=>0.4);d5 = Dict(:v_des=>27.0,:σ=>0.2)
	car_particles = [d1,d2,d3,d4,d5]

	particle_props = [(:v_des,10.,0.1,30.),(:σ,0.1,0.1,1.)]
	rec,rec_sim_pf,rec_sim_cem,roadway,plots_pf = viz_particle_dist(num_particles,num_cars,lane_place_array,
	    car_particles,particle_props)

	make_gif(plots_pf)

	#duration, fps, render_hist = make_gif(rec, rec_sim_pf,rec_sim_cem, roadway, 0.1)
	#film = roll(render_hist,fps=fps,duration=duration)
	#write("5car_ghost+pf+cem.gif",film)
"""
# Scenario 1: 2 cars in the same lane	

	num_particles = 100
	pos_vel_array_1 = [(30.,18.),(10.,12.)]
	lane_place_array = [pos_vel_array_1]
	num_cars = 2
	d1 = Dict(:v_des=>20.0,:σ=>0.1,:T=>1.5);d2 = Dict(:v_des=>10.0,:σ=>0.1,:T=>1.5)
	car_particles = [d1,d2]
	particle_props = [(:v_des,10.,0.1,30.),(:σ,0.1,0.1,1.),(:T,0.1,0.1,5.)]

	rec,rec_sim_pf,rec_sim_cem,roadway,plots_pf = viz_particle_dist(num_particles,num_cars,lane_place_array,
	    car_particles,particle_props)

	make_gif(plots_pf)
	#duration, fps, render_hist = make_gif(rec, rec_sim_pf,rec_sim_cem, roadway, 0.1)
	#film = roll(render_hist,fps=fps,duration=duration)
	#write("2car_ghost+pf+cem.gif",film)
