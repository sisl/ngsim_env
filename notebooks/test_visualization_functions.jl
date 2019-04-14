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
