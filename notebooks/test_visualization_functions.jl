@testset "ghost_overlay" begin
	pos_vel_array_1 = [(0.,10.)]
	pos_vel_array_2 = [(0.,0.)]
	pos_vel_array_3 = [(0.,0.),(10.,10.)]
	lane_place_array = [pos_vel_array_1,pos_vel_array_2,pos_vel_array_3]
	scene,roadway = init_place_cars(lane_place_array)
	d1 = Dict(:v_des=>10.0,:σ=>0.);d2 = Dict(:v_des=>10.0,:σ=>0.)
	d3 = Dict(:v_des=>10.0,:σ=>0.);d4 = Dict(:v_des=>10.0,:σ=>0.)
	car_particle_array = [d1,d2,d3,d4]

	rec = generate_truth_data(lane_place_array,car_particle_array)
	ghost_overlay = veh_overlay(rec.frames[100])
	@test length(ghost_overlay.sim_scene.entities) == 100 # 4 cars in scene but 100 elem container defined
end

@testset "animate_record" begin
	num_p = 100
	lane_place_array = [[(0.,10.)],[(0.,20.)],[(0.,15.)],[(0.,20.)],[(0.,20.)]]
	n_cars = 5
	d1 = Dict(:v_des=>10.0,:σ=>0.2);d2 = Dict(:v_des=>20.0,:σ=>0.3);d3 = Dict(:v_des=>15.0,:σ=>0.)
	d4 = Dict(:v_des=>18.0,:σ=>0.4);d5 = Dict(:v_des=>27.0,:σ=>0.2)
	car_particles = [d1,d2,d3,d4,d5]
	approach = "cem"

	particle_props = [(:v_des,10.,0.1,30.),(:σ,0.1,0.1,1.)]
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
	sim_rec = generate_truth_data(lane_place_array,car_particles_sim);

	duration, fps, render_hist = animate_record(rec, sim_rec, roadway, 0.1)
	@test duration == 10.
	@test fps == 10
end
