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

# Returns
- `array_of_particle_buckets::Array` Each element corresponds to a different car
Each element contains a dict with keys as IDM paramters and values as array with 
each element in that array being a different particle for that parameter
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

"""
    find_mean_particle_carwise(bucket_array::Array)

Find mean particle for each car

# Arguments
- `bucket_array::Array{Dict}` Array with each element corresponding
to a different car's associated particle of buckets. Each element is a
dict with keys as IDM parameters and value as array of particles

# Returns
- `carwise_mean_particle::Array{Dict}` Array with each element
corresponding to different car. Each element is a dict that 
contains the mean from the input bucket of particles
"""
function find_mean_particle_carwise(bucket_array::Array)
    n_cars = length(bucket_array)
    carwise_mean_particle = Array{Dict}(undef,n_cars)
    for i in 1:n_cars
        carwise_mean_particle[i] = find_mean_particle(bucket_array[i])
    end
    return carwise_mean_particle
end

"""
    zero_dict(keys::Array)
Create a dictionary using the provided keys with 0 associated values

# Arguments
- `keys:Array{Symbol}` An array containing the keys. Caution: these must be of type Symbol.
This is because the use case is working with IDM params, and these are of type Symbol
"""
function zero_dict(keys::Array)
    d = Dict{Symbol,Float64}()
    for k in keys
        d[k]=0.
    end
    return d
end

"""
	mean_dict(a::Array)
Find a dict which has the mean value of input dicts in `a`, which is an array of input dicts

Can possibly be done more elegantly using `merge!(+,a...)`

# Returns
- `b:Dict` Dict with same keys as the input dicts have, and value that is the mean
"""
function mean_dict(a::Array)
    params = collect(keys(a[1]))
    b = zero_dict(params)
    n_dicts = length(a)
    for i in 1:n_dicts
        b = merge(+,b,a[i])
    end
    for (k,v) in b
        b[k] = v/n_dicts
    end
    return b
end

"""
	compute_mean_dict(q::Array)
Find a dict which has the mean value of input dicts in `a`, which is an array of input dicts

Is alternate way to `mean_dict`. This function was written to avoid the loop and
merge all the dicts together in one fell swoop using `merge!`

See also: [`mean_dict`]

# Returns
- `n:Dict` Dict with same keys as the input dicts have, and value that is the mean
"""
function compute_mean_dict(q::Array)
    n = merge!(+,q...)
    for (k,v) in n
        n[k] = v/length(q)
    end
    return n
end

"""
    init_empty_array_dict(keys_array::Array,n::Int64)
Initialize a dictionary with empty array of given size `n::Int64` associated with given keys 
in `keys_array::Array` 
"""
function init_empty_array_dict(keys_array::Array,n::Int64)
    d = Dict{Symbol,Array}()
    for k in keys_array
        d[k]=Array{Float64}(undef,n)
    end
    return d
end

"""
    combine_array_dicts(q::Array)
Combine array of dictionaries `q::Array` with all elements having the same key
into a single dict with the same keys and associated value being all the values in the input array
of dicts combined into associated arrays

# Other functions used
- `init_empty_array_dic`
"""
function combine_array_dicts(q::Array)
    params = collect(keys(q[1]))
    num_params = length(params)
    num_vals = length(q)

    # Create an array with keys as given in params and associated value as an empty array of length num_vals
    d = init_empty_array_dict(params,num_vals)

    for i in 1:num_vals
        input_dict = q[i]
        for k in keys(input_dict)
            d[k][i] = input_dict[k]
        end
    end
    return d
end

"""
    plot_dict(d::Dict)
Plot the values of a dict separated by keys. Can't be tested so providing example below

# Example
A = Dict(:v_des=>20.,:s=>1.,:T=>0.1)
B= Dict(:v_des=>10.,:s=>2.,:T=>0.4)
C= Dict(:v_des=>40.,:s=>6.,:T=>0.7)
D= Dict(:v_des=>30.,:s=>5.,:T=>0.2)
q = [A,B,C,D]
d = combine_array_dicts(q)
plot_dict(d)
"""
function plot_dict(d::Dict)
    for k in keys(d)
        plot(d[k],label=k)
    end
    legend()
end

"""
    concat_symbols(a::Symbol,b::Symbol)
Concatenate symbols with underscore in the middle
"""
function concat_symbols(a::Symbol,b::Symbol)
    return Symbol(String(a)*String("_"*String(b)))
end

"""
Combine different experiment results into one

Different experiments result in the same values. We want to plot them all on one graph

# Arguments
- `names_symbols::Array` An array containing the experiment names as symbols eg: [:pf,:cem]
- `a::Array` Array containing dictionaries corresponding to the different experiments
eg: [Dict(:v_des=>20.,:s=>0.1),Dict(:v_des=>30.,:s=>0.4)]

# Returns
- `d::Dict` Dict with keys as the experiment results but now with exp name concatenated and
values as the associated value from the orginal experiments

# Other functions used
- `concat_symbols`
"""
function combine_exp_results_dicts(names_symbols::Array,a::Array)
    @assert length(names_symbols) == length(a)
    comb_dict = Dict{Symbol,Dict}()
    for i in 1:length(names_symbols)
        comb_dict[names_symbols[i]] = a[i]
    end

    d = Dict{Symbol,Array}()
    for k_comb in keys(comb_dict)
        for k_indiv in keys(comb_dict[k_comb])
            new_key = concat_symbols(k_indiv,k_comb)
            new_val = comb_dict[k_comb][k_indiv]
            d[new_key] = new_val
        end
    end
    return d
end
