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

# Plot both cem and pf in the same scatter plot
function plot_both_particles(p_set_mat_pf::Array{Float64,2},
			p_set_mat_cem::Array{Float64,2},true_params)
	# Check that number of params does not exceed 3
	@assert size(p_set_mat_cem,1) <= 3
	@assert size(p_set_mat_pf,1) <= 3
	@assert size(p_set_mat_pf,1) == size(p_set_mat_cem,1)
	plt = plot()
	# 2 parameter case	
	if size(p_set_mat_pf,1) == 2
		plt = scatter(p_set_mat_pf[1,:],p_set_mat_pf[2,:],
			label=["vanilla","cem","true"])
		scatter!(p_set_mat_cem[1,:],p_set_mat_cem[2,:],
			label=["vanilla","cem","true"])
		scatter!([true_params[1]],[true_params[2]],
			label=["vanilla","cem","true"])
	# 3 parameter case
	else
		plt = scatter(p_set_mat_pf[1,:],p_set_mat_pf[2,:],
				p_set_mat_pf[3,:],label=["vanilla","cem","true"])
		scatter!(p_set_mat_cem[1,:],p_set_mat_cem[2,:],
				p_set_mat_cem[3,:],label=["vanilla","cem","true"])
		scatter!([true_params[1]],[true_params[2]],[true_params[3]],
				label=["vanilla","cem","true"])
		#savefig(plt,"media/test.png")
	end
	return plt
end

function make_gif_allcars(plot_array::Array)
	# Every row of plot_array becomes one video
	@show "make_gif_allcars"
	for i in 1:size(plot_array,1)
		make_gif(plot_array[i,:],filename = "car_$(i).mp4")
	end
	return nothing
end

function viz_particle_distribution(num_p::Int64,n_cars::Int64,lane_place_array::Array,
        car_particles::Array,particle_props::Array)
    scene,roadway = init_place_cars(lane_place_array)
    rec = generate_truth_data(lane_place_array,car_particles)
    f_end_num = length(rec.frames)

	# Array of particle sets. Each element corresponds to different car    
    bucket_array_pf = initialize_carwise_particle_buckets(n_cars,num_p,particle_props)
    bucket_array_cem = initialize_carwise_particle_buckets(n_cars,num_p,particle_props)

	# Array to store the plots
	plot_array = Array{Any}(undef,n_cars,f_end_num-1)

	# Loop over the entire sequence of observations
    for t in 1:f_end_num-1
        f = rec.frames[f_end_num - t + 1]
		# Loop over the sequence of cars
        for car_id in 1:n_cars
		# Access the observation i.e what truly the car driving looked like
            trupos = rec.frames[f_end_num-t].entities[car_id].state.posF.s

		# Do the particle set update using vanilla particle filtering
            old_p_set_dict_pf = bucket_array_pf[car_id]
            new_p_set_dict_pf = update_p_one_step(roadway,f,trupos,old_p_set_dict_pf,
                car_id=car_id,approach="pf")
            bucket_array_pf[car_id] = new_p_set_dict_pf

		# Do the particle set update using cross entropy method
            old_p_set_dict_cem = bucket_array_cem[car_id]
            new_p_set_dict_cem = update_p_one_step(roadway,f,trupos,old_p_set_dict_cem,
                car_id=car_id,approach="cem")
            bucket_array_cem[car_id] = new_p_set_dict_cem


		true_param_dict = car_particles[car_id]
		trueparams = trueparam_dict2array(true_param_dict)
		particles_pf = to_particleMatrix(new_p_set_dict_pf)
		particles_cem = to_particleMatrix(new_p_set_dict_cem)

		# Store the particle scatter in the plot_array
		plot_array[car_id,t] = plot_both_particles(
				particles_pf,particles_cem,trueparams)
        end
    end  
    
    # Estimation of parameters ends here. Now generate simulation
    car_particles_pf = find_mean_particle_carwise(bucket_array_pf)
    sim_rec_pf = generate_truth_data(lane_place_array,car_particles_pf)
    
    # -------------CEM particle filter--------------
    
    # Estimation of parameters ends here. Now generate simulation
    car_particles_cem = find_mean_particle_carwise(bucket_array_cem)
    sim_rec_cem = generate_truth_data(lane_place_array,car_particles_cem)
    
    return rec,sim_rec_pf,sim_rec_cem,roadway, plot_array
end # End of viz function defined

# Run experiments
# Scenario 2: 5 cars in adjacent lanes

	num_particles = 100
	lane_place_array = [[(0.,10.)],[(0.,20.)],[(0.,15.)],[(0.,20.)],[(0.,20.)]]
	num_cars = 5
	d1 = Dict(:v_des=>10.0,:σ=>0.2);d2 = Dict(:v_des=>20.0,:σ=>0.3);d3 = Dict(:v_des=>15.0,:σ=>0.)
	d4 = Dict(:v_des=>18.0,:σ=>0.4);d5 = Dict(:v_des=>27.0,:σ=>0.2)
	car_particles = [d1,d2,d3,d4,d5]

	particle_props = [(:v_des,10.,0.1,30.),(:σ,0.1,0.1,1.)]
	rec,rec_sim_pf,rec_sim_cem,roadway,plot_array = viz_particle_distribution(num_particles,num_cars,lane_place_array,
	    car_particles,particle_props)

	make_gif_allcars(plot_array)

	#duration, fps, render_hist = make_gif(rec, rec_sim_pf,rec_sim_cem, roadway, 0.1)
	#film = roll(render_hist,fps=fps,duration=duration)
	#write("5car_ghost+pf+cem.gif",film)

# Scenario 1: 2 cars in the same lane	
"""
	num_particles = 100
	pos_vel_array_1 = [(30.,18.),(10.,12.)]
	lane_place_array = [pos_vel_array_1]
	num_cars = 2
	d1 = Dict(:v_des=>20.0,:σ=>0.1,:T=>1.5);d2 = Dict(:v_des=>10.0,:σ=>0.1,:T=>1.5)
	car_particles = [d1,d2]
	particle_props = [(:v_des,10.,0.1,30.),(:σ,0.1,0.1,1.),(:T,0.1,0.1,5.)]

	rec,rec_sim_pf,rec_sim_cem,roadway,plot_array = viz_particle_distribution(num_particles,num_cars,lane_place_array,
	    car_particles,particle_props)

	make_gif_allcars(plot_array)
	#duration, fps, render_hist = make_gif(rec, rec_sim_pf,rec_sim_cem, roadway, 0.1)
	#film = roll(render_hist,fps=fps,duration=duration)
	#write("2car_ghost+pf+cem.gif",film)
"""
