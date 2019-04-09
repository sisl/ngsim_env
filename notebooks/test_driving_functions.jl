# Test the driving functions

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
