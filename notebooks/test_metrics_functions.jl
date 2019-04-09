# Test the metrics functions

@testset "calc_rmse_pos" begin
	# Generate truth data
	pos_vel_array_1 = [(0.,10.)]
	pos_vel_array_2 = [(0.,10.)]
	pos_vel_array_3 = [(0.,10.)]
	lane_place_array = [pos_vel_array_1,pos_vel_array_2,pos_vel_array_3]
	scene,roadway = init_place_cars(lane_place_array)
	d1 = Dict(:v_des=>10.0,:σ=>0.);d2 = Dict(:v_des=>10.0,:σ=>0.)
	d3 = Dict(:v_des=>10.0,:σ=>0.)
	car_particle_array = [d1,d2,d3]

	truerec = generate_truth_data(lane_place_array,car_particle_array)

	# Generate simulated data
	pos_vel_array_1 = [(0.,10.)]
	pos_vel_array_2 = [(0.,10.)]
	pos_vel_array_3 = [(0.,10.)]
	lane_place_array = [pos_vel_array_1,pos_vel_array_2,pos_vel_array_3]
	scene,roadway = init_place_cars(lane_place_array)
	d1 = Dict(:v_des=>10.0,:σ=>0.);d2 = Dict(:v_des=>10.0,:σ=>0.)
	d3 = Dict(:v_des=>11.0,:σ=>0.)
	car_particle_array = [d1,d2,d3]

	simrec = generate_truth_data(lane_place_array,car_particle_array)

	test_rmse_pos = calc_rmse_pos(truerec,simrec,num_cars=3)
	
	@test length(test_rmse_pos) == 100
	@test test_rmse_pos[20] > test_rmse_pos[5]

end

@testset "particle_difference" begin
	A = Dict(:T=>0.4,:v_des => 20.)
	B = Dict(:v_des => 20., :T=>0.4)
	@test particle_difference(A,B) == 0.
	
	# Would like to be able to test something like this
	# Should throw an error as keys aren't the same for both dicts
	#A = Dict(:T=>0.4,:v_des => 20.)
	#B = Dict(:v_des => 20., :p=>0.4)
	#@test_throws ErrorException particle_difference(A,B) == 0.
end

@testset "particle_difference_paramwise" begin
	a = Dict(:v_des=>25.,:T=>0.5)
	b = Dict(:v_des=>40.,:T=>2.5)
	c = particle_difference_paramwise(a,b)
	@test c[:v_des] == 15.
	@test c[:T] == 2.
end
