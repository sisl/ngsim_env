# Test the filtering functions
@testset "compute_particle_likelihoods" begin
	
	v_array = [10.,15.,20.,25.,30.]
	num_p = length(v_array)
	sig_array = [0.1,0.1,0.1,0.1,0.1]	
	p_set_dict = Dict(:v_des=>v_array,:σ=>sig_array)
	scene,roadway = init_scene_roadway([0.0])
	trupos = hallucinate_a_step(roadway,scene,Dict(:v_des=>25.0,:σ=>0.0),car_id=1)
	lkhd_vec,p_mat,params = compute_particle_likelihoods(roadway,scene,trupos,p_set_dict,car_id=1)

	@test length(lkhd_vec) == num_p
	@test length(params) == 2
	@test size(p_mat)[1] == 2
	@test size(p_mat)[2] == 5
	@test any(isnan,lkhd_vec) == false #Gets triggered when wrong car_id called
	# For example only 1 car on road but you say car_id = 2
end

# Occasionally fails saying Cholesky factorization failes
@testset "update_p_one_step" begin
	num_p = 5
	car_pos = [0.,0.,0.]
	n_cars = length(car_pos)
	scene,roadway = init_scene_roadway(car_pos)
	d1 = Dict(:v_des=>10.0,:σ=>0.);d2 = Dict(:v_des=>10.0,:σ=>0.);d3 = Dict(:v_des=>10.0,:σ=>0.)
	car_particles = [d1,d2,d3]
	car_vel_array = [10.,0.,0.]
	rec = generate_ground_truth(car_pos,car_particles,car_vel_array=car_vel_array,n_steps=100)
	
	# Tests the cem approach
	for i in 1:n_cars
	    trupos = rec.frames[100].entities[i].state.posF.s
	    p_set_dict = gen_test_particles(num_p)
	    p_set_new = update_p_one_step(roadway,scene,trupos,p_set_dict,car_id=i,approach="cem",elite_fraction_percent=60)
	    @test length(keys(p_set_new)) == 2
	    @test length(p_set_new[:v_des]) == 5
	end

	# Test the pf approach
	for i in 1:n_cars
	    trupos = rec.frames[100].entities[i].state.posF.s
	    p_set_dict = gen_test_particles(num_p)
	    p_set_new = update_p_one_step(roadway,scene,trupos,p_set_dict,car_id=i,approach="pf",elite_fraction_percent=60)
	    @test length(keys(p_set_new)) == 2
	    @test length(p_set_new[:v_des]) == 5
	end
end

@testset "filter_particles_over_trajectory" begin
	# 5 cars adjacent lanes scenario with 2 parameters
	num_particles = 100
	lane_place_array = [[(0.,10.)],[(0.,20.)],[(0.,15.)],[(0.,20.)],[(0.,20.)]]
	num_cars = 5
	d1 = Dict(:v_des=>10.0,:σ=>0.2);d2 = Dict(:v_des=>20.0,:σ=>0.3);d3 = Dict(:v_des=>15.0,:σ=>0.)
	d4 = Dict(:v_des=>18.0,:σ=>0.4);d5 = Dict(:v_des=>27.0,:σ=>0.2)
	car_particles = [d1,d2,d3,d4,d5]

	particle_props = [(:v_des,10.,0.1,30.),(:σ,0.1,0.1,1.)]
	bucket_array = filter_particles_over_trajectory(num_particles,num_cars,lane_place_array,car_particles,particle_props,approach="cem")
	@test length(bucket_array) == 5
	@test length(keys(bucket_array[1])) == 2
end

@testset "capture_filtering_progress" begin
	# 5 cars adjacent lanes scenario with 2 parameters
	num_particles = 100
	lane_place_array = [[(0.,10.)],[(0.,20.)],[(0.,15.)],[(0.,20.)],[(0.,20.)]]
	num_cars = 5
	d1 = Dict(:v_des=>10.0,:σ=>0.2);d2 = Dict(:v_des=>20.0,:σ=>0.3);d3 = Dict(:v_des=>15.0,:σ=>0.)
	d4 = Dict(:v_des=>18.0,:σ=>0.4);d5 = Dict(:v_des=>27.0,:σ=>0.2)
	car_particles = [d1,d2,d3,d4,d5]

	particle_props = [(:v_des,10.,0.1,30.),(:σ,0.1,0.1,1.)]
	rmse_array_pf = capture_filtering_progress(num_particles,num_cars,lane_place_array,
	    car_particles,particle_props,approach="pf")
	rmse_array_cem = capture_filtering_progress(num_particles,num_cars,lane_place_array,
	    car_particles,particle_props,approach="cem")
	@test length(rmse_array_pf) == 99
	@test length(rmse_array_cem) == 99
end

@testset "capture_filtering_progress_paramwise" begin
	# 5 cars adjacent lanes scenario with 2 parameters
	num_particles = 100
	lane_place_array = [[(0.,10.)],[(0.,20.)],[(0.,15.)],[(0.,20.)],[(0.,20.)]]
	num_cars = 5
	d1 = Dict(:v_des=>10.0,:σ=>0.2);d2 = Dict(:v_des=>20.0,:σ=>0.3);d3 = Dict(:v_des=>15.0,:σ=>0.)
	d4 = Dict(:v_des=>18.0,:σ=>0.4);d5 = Dict(:v_des=>27.0,:σ=>0.2)
	car_particles = [d1,d2,d3,d4,d5]

	particle_props = [(:v_des,10.,0.1,30.),(:σ,0.1,0.1,1.)]
	rmse_array_pf = capture_filtering_progress_paramwise(num_particles,num_cars,lane_place_array,
	    car_particles,particle_props,approach="pf")
	rmse_array_cem = capture_filtering_progress_paramwise(num_particles,num_cars,lane_place_array,
	    car_particles,particle_props,approach="cem")
	@test length(rmse_array_pf) == 99
	@test length(rmse_array_cem) == 99
end
