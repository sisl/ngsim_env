"""
gen_test_particles: Helper function for testing stuff. Generates dictionary with
keys as params and values as array of particles
"""
function gen_test_particles(num_p::Int64)
    # start:step:end and number of particles are the inputs to sample
    v_particles = sample(10.0:1.0:30.0,num_p)
    sig_particles = sample(0.1:0.1:1.0,num_p)
    p_set_dict = Dict(:v_des=>v_particles,:σ=>sig_particles)
    return p_set_dict
end

"""
initialize_particles: A more general particle initializer than `gen_test_particles
`gen_test_particles` was hard coded to work only with `v_des` and `sigma`.
Also had hard coded range to sample from

------Args that need explanation:
`input`: Array with each element corresponding to a different parameter.
Each element is a tuple with 4 elements. These are
symbol with param name, start value to sample from, step, end value to sample

--------Returns:
- `p_set_dict` A dictionary with keys as parameters and values as arrays with each
element being a different particle

"""
function initialize_particles(input::Array,num_particles::Int64)
    p_set_dict = Dict{Symbol,Array}()
    for i in 1:length(input)
        p_set_dict[input[i][1]] = sample(input[i][2]:input[i][3]:input[i][4],
            num_particles)
    end
    return p_set_dict
end

"""
to_matrix_form: Return information and more workable form for initial particle set

---------Arguments:
`p_set_dict` Dictionary with parameters of IDM as keys and associated value as array of particles

---------Returned things that need explanation:
`p_mat` Matrix with each row corresponding to a different parameter of IDM, and each column to a diff particle
`params` Array of parameters eg: [:v_des,:σ]
`vec_val_vec` Array with each element being another array. This array contains the values i.e diff particles
"""
function to_matrix_form(p_set_dict)
    # Get the number of particles
    # Extract keys (i.e. params) and corresponding array of values
    num_p = -100
    num_params = length(keys(p_set_dict))
    
    params = Array{Symbol}(undef,num_params,1)
    vec_val_vec = Array{Array}(undef,num_params,1) #Array containing associated values for each key
    for (kk,kv) in enumerate(p_set_dict)
        num_p = length(kv[2])
        params[kk] = kv[1]
        vec_val_vec[kk] = kv[2]
    end
    
    # Create a matrix with different rows being different parameters and diff cols being diff particles
    p_mat = hcat(vec_val_vec...)'
    
    return p_mat, params, vec_val_vec
end

"""
Create a new dictionary with param and associated particle value array
"""
function to_dict_form(params,new_p_mat)
    num_params = length(params)
    new_p_set_dict = Dict()
    for k in 1:num_params
        new_p_set_dict[params[k]] = new_p_mat[k,:]
    end
    return new_p_set_dict
end

"""
Initialize an array with each element being the associated bucket of particles
for each car

------Functions called: `gen_test_particles`
"""
function init_car_particle_buckets(n_cars::Int64,num_particles::Int64)
    array_of_particle_buckets = Array{Dict}(undef,n_cars)
    for i in 1:n_cars
        array_of_particle_buckets[i] = gen_test_particles(num_particles)
    end
    return array_of_particle_buckets
end

"""
	initialize_carwise_particle_buckets(n_cars::Int64,num_particles::Int64)

More general particle buckets initialization associated with every car.
`init_car_particle_buckets` used `gen_test_particles` and hence was limited
"""
function initialize_carwise_particle_buckets(n_cars::Int64,num_particles::Int64,input::Array)
    array_of_particle_buckets = Array{Dict}(undef,n_cars)
    for i in 1:n_cars
        array_of_particle_buckets[i] = initialize_particles(input,num_particles)
    end
    return array_of_particle_buckets
end

"""
	print_buckets_mean(bucket_array::Array)

Print the mean particle values carwise i.e. car1 printed in first row, car 2 in second
Each element is printed as an array with the mean values as the elements

---------Argument
- `bucket_array` This is an array with each element being a dictionary.
This dictionary has key as parameters of IDM and values as array. Each element in this
array is a different particle
"""
function print_buckets_mean(bucket_array::Array)
    for i in 1:length(bucket_array)
        params = []
        for (k,v) in bucket_array[i]
            push!(params,mean(v))
        end
        @show params
    end
end

"""
    find_mean_particle(p_set_dict::Dict)

# Arguments
- `p_set_dict`: Key is the parameter name. Value is an array containing all the particles

# Returns
- `mean_particle`: Is a dict with key as parameter name and value as the mean of all particles
"""
function find_mean_particle(p_set_dict::Dict)
    mean_particle = Dict()
    for (k,v) in p_set_dict
        mean_particle[k] = mean(v)
    end
    return mean_particle
end
