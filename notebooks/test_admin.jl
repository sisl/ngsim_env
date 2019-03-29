"""
Tests pertaining to utilities such as conversion to and from dict to matrix form
"""
include("admin.jl")
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

@testset "gen_test_particles" begin
	p_set_dict = gen_test_particles(5)
	@test length(keys(p_set_dict)) == 2
	@test length(p_set_dict[:v_des]) == 5
end
