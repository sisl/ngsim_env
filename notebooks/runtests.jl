# The file that is used to run the tests. `julia runtests.jl`

using Test
using Distributions
using AutomotiveDrivingModels
include("admin_functions.jl")
include("driving_functions.jl")
include("filtering_functions.jl")

@testset "gen_test_particles" begin
	p_set_dict = gen_test_particles(5)
	@test length(keys(p_set_dict)) == 2
	@test length(p_set_dict[:v_des]) == 5
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

@testset "hallucinate_a_step" begin
	scene,roadway = init_scene_roadway([0.0,10.0,20.0])
	particle = Dict(:v_des=>25.0,:σ=>0.5)
	@test isapprox(hallucinate_a_step(roadway,scene,particle,car_id=1),0.02,atol=0.1)
	@test isapprox(hallucinate_a_step(roadway,scene,particle,car_id=2),10.02,atol=0.1)
	@test isapprox(hallucinate_a_step(roadway,scene,particle,car_id=3),20.02,atol=0.15)
end

@testset "compute_particle_likelihoods" begin
	num_p = 5
	p_set_dict = gen_test_particles(num_p)
	scene,roadway = init_scene_roadway([0.0,10.0,20.0])
	trupos = hallucinate_a_step(roadway,scene,Dict(:v_des=>25.0,:σ=>0.0),car_id=2)
	lkhd_vec,p_mat,params = compute_particle_likelihoods(roadway,scene,trupos,p_set_dict,car_id=2)

	@test length(lkhd_vec) == num_p
	@test length(params) == 2
	@test size(p_mat)[1] == 2
	@test size(p_mat)[2] == 5
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
