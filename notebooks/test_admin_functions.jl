# This file tests the admin functions

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

@testset "find_mean_particle" begin
	p_set_dict = Dict(:v_des=>[10,20,30,40],:sigma=>[0.1,0.2,0.3,0.4])
	mean_p = find_mean_particle(p_set_dict)
	@test mean_p[:v_des] == 25.0
	@test mean_p[:sigma] == 0.25
end

@testset "zero_dict" begin
	n = zero_dict([:v_des,:σ,:T,:s])
	@test length(keys(n)) == 4
	@test n[:v_des] == 0.
	@test n[:T] == 0.
	@test n[:σ] == 0.
end

@testset "mean_dict" begin
	a = [Dict(:v_des=>10.,:T=>1.),Dict(:v_des=>20.,:T=>2.),Dict(:v_des=>30.,:T=>3.)]
	b = [Dict(:v_des=>15.,:T=>2.),Dict(:v_des=>30.,:T=>4.),Dict(:v_des=>45.,:T=>6.)]
	c = mean_dict(a)
	d = mean_dict(b)
	@test c[:v_des] == 20.
	@test c[:T] == 2.
	@test d[:v_des] == 30.
	@test d[:T] == 4.
end

@testset "compute_mean_dict" begin
	a = Dict(:s=>1.,:T=>2.)
	b = Dict(:s=>3.,:T=>7.)
	c = Dict(:s=>2.,:T=>5.)
	q = [a,b,c]
	n = compute_mean_dict(q)
	@test isapprox(n[:T],4.66,atol=0.01)
	@test n[:s] == 2.
end

@testset "init_empty_array_dict" begin
	keys_array = [:v_des,:T,:s]
	n = 4
	d = init_empty_array_dict(keys_array,n)
	@test length(d[:v_des]) == 4
	@test length(d[:s]) == 4
	@test length(d[:T]) == 4
end

@testset "combine_array_dicts" begin
	A = Dict(:v_des=>20.,:s=>1.,:T=>0.1)
	B= Dict(:v_des=>10.,:s=>2.,:T=>0.4)
	C= Dict(:v_des=>40.,:s=>6.,:T=>0.7)
	D= Dict(:v_des=>30.,:s=>5.,:T=>0.2)
	q = [A,B,C,D]
	d = combine_array_dicts(q)
	@test d[:v_des][2] == 10.
	@test d[:T][4] == 0.2
	@test d[:T][1] == 0.1
	@test d[:s][4] == 5.
end

@testset "concat_symbols" begin
	s = concat_symbols(:T,:cem)
	@test s == :T_cem 
	s = concat_symbols(:v_des,:pf)
	@test s == :v_des_pf
end

@testset "combine_exp_results_dicts" begin
	p = Dict(:v_des=>[20.,30.],:T=>[1.,2.],:s=>[4.,6.])
	q = Dict(:v_des=>[50.,60.],:T=>[11.,12.],:s=>[14.,8.])

	a = [p,q]

	names_symbols = [:pf,:cem]

	res = combine_exp_results_dicts(names_symbols, a)
	@test res[:s_cem][1] == 14.
	@test res[:s_cem][2] == 8.
	@test res[:v_des_cem][1] == 50.
	@test res[:v_des_cem][2] == 60.
	@test res[:v_des_pf][1] == 20.
	@test res[:v_des_pf][2] == 30.
end
