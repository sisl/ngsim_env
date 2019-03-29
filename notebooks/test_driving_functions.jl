"""
Tests for functions pertaining to driving such as scene init, hallucination
"""
include("driving_functions.jl")

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
