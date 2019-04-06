# The file that is used to run the tests. `julia runtests.jl`

using Test
using Distributions
using AutomotiveDrivingModels
include("admin_functions.jl")
include("driving_functions.jl")
include("filtering_functions.jl")

#-------------------test_admin_functions-------------------------
@testset "gen_test_particles" begin
	p_set_dict = gen_test_particles(5)
	@test length(keys(p_set_dict)) == 2
	@test length(p_set_dict[:v_des]) == 5
end

@testset "initialize_particles" begin
	input = [(:v_des,10.,0.1,30.),(:σ,0.1,0.1,1.),(:T,0.1,0.1,5.)]
	p = initialize_particles(input,5)
	@test length(keys(p)) == 3
	@test length(p[:v_des]) == 5
end

@testset "to_matrix_form" begin
	num_p = 5
	p_set_dict = gen_test_particles(num_p)
	p_mat, params, vec_val_vec = to_matrix_form(p_set_dict);
	@test params[1] == :v_des
	@test params[2] == :σ
	@test size(p_mat)[1] == 2
	@test size(p_mat)[2] == 5
	@test length(vec_val_vec) == 2
	@test length(vec_val_vec[1]) == 5
	@test p_mat[1,2] == p_set_dict[:v_des][2]
	@test p_mat[2,3] == p_set_dict[:σ][3]
end

@testset "to_dict_form" begin
	params = [:v_des,:σ]
	new_p_mat = [17.0 26.0 24.0 19.0 29.0; 0.1 0.1 0.7 0.4 0.1]
	new_p_set_dict = to_dict_form(params,new_p_mat)
	@test length(keys(new_p_set_dict))==2
	@test new_p_set_dict[:v_des][1] == 17.0
	@test new_p_set_dict[:σ][1] == 0.1
end

@testset "init_car_particle_buckets" begin
	bucket_array = init_car_particle_buckets(3,5)
	@test length(keys(bucket_array[1])) == 2
	@test length(bucket_array[2][:σ]) == 5
end

@testset "initialize_carwise_particle_buckets" begin
	input = [(:v_des,10.,0.1,30.),(:σ,0.1,0.1,1.),(:T,0.1,0.1,5.)]
	bucket_array = initialize_carwise_particle_buckets(3,5,input)
	@test length(keys(bucket_array[1])) == 3
	@test length(bucket_array[2][:σ]) == 5
end

#-------------------test_driving_functions-------------------------
@testset "init_scene_roadway" begin
	scene,road = init_scene_roadway([0.,10.,20.,30.],car_vel_array=[0.,10.,20.,0.])
	@test scene[1].state.posF.s==0.0
	@test scene[2].state.posF.s==10.0
	@test scene[3].state.posF.s==20.0
	@test scene[1].state.posF.roadind.tag.lane == 1
	@test scene[2].state.posF.roadind.tag.lane == 2
	@test scene[3].state.posF.roadind.tag.lane == 3
	@test scene[1].id == 1
	@test scene[2].id == 2
	@test scene[3].id == 3
	@test scene[3].state.v == 20.
	@test scene[4].state.v == 0.
end

@testset "init_place_cars" begin
	pos_vel_array_1 = [(10.,30.),(15.,0.),(20.,0.)]
	pos_vel_array_2 = [(10.,0.),(15.,0.),(20.,20.)]
	pos_vel_array_3 = [(15.,0.),(25.,10.),(30.,0.)]
	lane_place_array = [pos_vel_array_1,pos_vel_array_2,pos_vel_array_3]
	scene,roadway = init_place_cars(lane_place_array)
	for i in 1:9
	    @test scene.entities[i].id == i
	end
	for i in 1:3
	    @test scene.entities[i*3].state.posF.roadind.tag.lane == i
	end
	@test scene.entities[1].state.v == 30.
	@test scene.entities[6].state.v == 20.
	@test scene.entities[8].state.v == 10.
	@test scene.entities[1].state.posF.s == 10.
	@test scene.entities[2].state.posF.s == 15.
	@test scene.entities[5].state.posF.s == 15.
	@test scene.entities[8].state.posF.s == 25.
end

@testset "generate_ground_truth" begin
	car_pos = [0.,0.,0.]
	scene,roadway = init_scene_roadway(car_pos) # Not required for test per se but required for rendering
	d1 = Dict(:v_des=>10.0,:σ=>0.)
	d2 = Dict(:v_des=>10.0,:σ=>0.)
	d3 = Dict(:v_des=>10.0,:σ=>0.)
	car_particles = [d1,d2,d3]
	car_vel_array = [10.,0.,0.]
	rec = generate_ground_truth(car_pos,car_particles,car_vel_array=car_vel_array,n_steps=100)
	@test isapprox(rec.frames[1].entities[1].state.posF.s,100.0)
	@test isapprox(rec.frames[1].entities[2].state.posF.s, 81.9,atol=0.1)
end

@testset "generate_truth_data" begin
	pos_vel_array_1 = [(0.,10.)]
	pos_vel_array_2 = [(0.,0.)]
	pos_vel_array_3 = [(0.,0.),(10.,10.)]
	lane_place_array = [pos_vel_array_1,pos_vel_array_2,pos_vel_array_3]
	scene,roadway = init_place_cars(lane_place_array)
	d1 = Dict(:v_des=>10.0,:σ=>0.);d2 = Dict(:v_des=>10.0,:σ=>0.)
	d3 = Dict(:v_des=>10.0,:σ=>0.);d4 = Dict(:v_des=>10.0,:σ=>0.)
	car_particle_array = [d1,d2,d3,d4]

	rec = generate_truth_data(lane_place_array,car_particle_array)
	@test isapprox(rec.frames[1].entities[1].state.posF.s,100.0,atol=0.1)
	@test isapprox(rec.frames[1].entities[2].state.posF.s, 81.9,atol=0.1)
	@test isapprox(rec.frames[1].entities[3].state.posF.s, 72.2,atol=0.1)
	@test isapprox(rec.frames[1].entities[4].state.posF.s, 110.0,atol=0.1)
end

@testset "hallucinate_a_step" begin	
	# Run for multiple CARS	
	scene,roadway = init_scene_roadway([0.0,10.0,20.0])
	particle = Dict(:v_des=>25.0,:σ=>0.5)
	@test isapprox(hallucinate_a_step(roadway,scene,particle,car_id=1),0.02,atol=0.1)
	@test isapprox(hallucinate_a_step(roadway,scene,particle,car_id=2),10.02,atol=0.1)
	@test isapprox(hallucinate_a_step(roadway,scene,particle,car_id=3),20.02,atol=0.15)

	# Run for multiple PARTICLES
	scene,roadway = init_scene_roadway([0.0,10.0,20.0])
	for i in 1:5
		particle = Dict(:v_des=>25.0,:σ=>0.5)
		@test isapprox(hallucinate_a_step(roadway,scene,particle,car_id=1),0.02,atol=0.1)
	end

	# Hallucinate using the new way of car particle bucket generation i.e more than 2 params
	scene,roadway = init_scene_roadway([0.])
	input = [(:T,0.1,0.1,10.),(:v_des,10.,0.1,30.),(:σ,0.1,0.1,1.)]
	bucket_array = initialize_carwise_particle_buckets(1,5,input)
		
	for i in 1:length(bucket_array)
		p_set_dict = bucket_array[i]
		p_mat, params, vec_val_vec = to_matrix_form(p_set_dict)
		num_params=size(p_mat)[1]
		num_p = size(p_mat)[2]		
		for i in 1:num_p
			# Create dict version for a single particle
        		p_dict = Dict()
	        	for j in 1:num_params
        		    p_dict[params[j]]=vec_val_vec[j][i]
        		end
	
			hallucinate_a_step(roadway,scene,p_dict,car_id=1)
		end
	end
end

#-------------------test_filtering_functions-------------------------
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
	    p_set_new = update_p_one_step(roadway,scene,trupos,p_set_dict,car_id=i,approach="cem",elite_fraction_percent=60)
	    @test length(keys(p_set_new)) == 2
	    @test length(p_set_new[:v_des]) == 5
	end
end

# Temporarily keeping here for running filtering over trajectory
num_particles = 100
pos_vel_array_1 = [(30.,15.),(10.,15.)]
lane_place_array = [pos_vel_array_1]
num_cars = 2
d1 = Dict(:v_des=>20.0,:σ=>0.1);d2 = Dict(:v_des=>10.0,:σ=>0.1)
car_particles = [d1,d2]
particle_props = [(:v_des,10.,0.1,30.),(:σ,0.1,0.1,1.)]
filter_trajectory(num_particles,num_cars,lane_place_array,car_particle_array,particle_props,approach="pf")
