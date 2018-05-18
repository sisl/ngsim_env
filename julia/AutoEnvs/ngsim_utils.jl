export 
    index_ngsim_trajectory,
    load_ngsim_trajdatas,
    sample_trajdata_vehicle,
    sample_multiple_trajdata_vehicle,
    build_feature_extractor,
    max_n_objects,
    fill_infos_cache,
    stack_tensor_dict_list,
    keep_vehicle_subset!

#=
Description:
    Creates an index (easily accessible set of information) for an ngsim trajdata.
    The index is specifically a dictionary mapping vehicle ids to a dictionary 
    of metadata, for example the first and last timestep where that vehicle is 
    present in the trajdata.

Args:
    - filepath: filepath to trajdata to load 
    - minlength: minimum trajectory length to include
    - vebose: print out progress

Returns:
    - index: index of the trajdata
=#
function index_ngsim_trajectory(
        filepath::String; 
        minlength::Int=100,
        offset::Int=500,
        verbose::Int=1)
    # setup
    index = Dict()
    trajdata = load_trajdata(filepath)
    n_frames = nframes(trajdata)
    scene_length = maximum(n_objects_in_frame(trajdata, i) for i in 1 : n_frames)
    scene = Scene(scene_length)
    prev, cur = Set(), Set()

    # iterate each frame collecting info about the vehicles
    for frame in offset : n_frames - offset
        if verbose > 0
            print("\rframe $(frame) / $(n_frames - offset)")
        end
        cur = Set()
        get!(scene, trajdata, frame)

        # add all the vehicles to the current set
        for veh in scene
            push!(cur, veh.id)
            # insert previously unseen vehicles into the index
            if !in(veh.id, prev)
                index[veh.id] = Dict("ts"=>frame)
            end
        end

        # find vehicles in the previous but not the current frame
        missing = setdiff(prev, cur)
        for id in missing
            # set the final frame for all these vehicles
            index[id]["te"] = frame - 1
        end

        # step forward
        prev = cur
    end

    # at this point, any ids in cur are in the last frame, so add them in 
    for id in cur
        index[id]["te"] = n_frames - offset
    end

    # postprocess to remove undesirable trajectories
    for (vehid, infos) in index
        # check for start and end frames 
        if !in("ts", keys(infos)) || !in("te", keys(infos))
            if verbose > 0
                println("delete vehid $(vehid) for missing keys")
            end
            delete!(index, vehid)

        # check for start and end frames greater than minlength
        elseif infos["te"] - infos["ts"] < minlength
            if verbose > 0
                println("delete vehid $(vehid) for below minlength")
            end
            delete!(index, vehid)
        end
    end

    return index
end

#=
Description:
    Loads trajdatas and metadata used for sampling individual trajectories

Args:
    - filepaths: list of filepaths to individual trajdatas

Returns:
    - trajdatas: list of trajdata objects, each to a timeperiod of NGSIM
    - trajinfos: list of dictionaries providing metadata for sampling
        each dictionary has 
            key = id of a vehicle in the trajdata
            value = first and last timestep in trajdata of vehicle
=#
function load_ngsim_trajdatas(filepaths; minlength::Int=100)
    # check that indexes exist for the relevant trajdatas
    # if they are missing, create the index
    # the index is just a collection of metadata that is saved with the 
    # trajdatas to allow for a more efficient environment implementation
    indexes_filepaths = [replace(f, ".txt", "-index-$(minlength).jld") for f in filepaths]
    indexes = Dict[]
    for (i, index_filepath) in enumerate(indexes_filepaths)
        # check if index already created
        # if so, load it
        # if not, create and save it
        if !isfile(index_filepath)
            index = index_ngsim_trajectory(filepaths[i], minlength=minlength)
            JLD.save(index_filepath, "index", index)
            # write this information to an hdf5 file for use in python
            # the reason we write two files is that it's convenient to load 
            # via jld a dictionary that can be used directly
            ids_filepath = replace(index_filepath, ".jld", "-ids.h5")
            ids = convert(Array{Int}, collect(keys(index)))
            ts = Int[]
            te = Int[]
            for id in ids
                push!(ts, index[id]["ts"])
                push!(te, index[id]["te"])
            end
            h5open(ids_filepath, "w") do file
                write(file, "ids", ids)  
                write(file, "ts", ts)
                write(file, "te", te)
            end
        else
            index = JLD.load(index_filepath)["index"]
        end

        # load index
        push!(indexes, index)
    end

    # load trajdatas
    trajdatas = Records.ListRecord[]
    roadways = Roadway[]
    for filepath in filepaths
        trajdata = load_trajdata(filepath)
        push!(trajdatas, trajdata)
        roadway = get_corresponding_roadway(filepath)
        push!(roadways, roadway)
    end

    return trajdatas, indexes, roadways
end

#=
Description:
    Sample a vehicle to imitate

Args:
    - trajinfos: the metadata list of dictionaries

Returns:
    - traj_idx: index of NGSIM trajdatas
    - egoid: id of ego vehicle
    - ts: start timestep for vehicle 
    - te: end timestep for vehicle
=#
function sample_trajdata_vehicle(
        trajinfos, 
        offset::Int=0,
        traj_idx::Union{Void,Int}=nothing,
        egoid::Union{Void,Int}=nothing,
        start::Union{Void,Int}=nothing)
    if traj_idx == nothing || egoid == nothing || start == nothing
        traj_idx = rand(1:length(trajinfos))
        egoid = rand(collect(keys(trajinfos[traj_idx])))
        ts = trajinfos[traj_idx][egoid]["ts"]
        te = trajinfos[traj_idx][egoid]["te"]
        ts = rand(ts:te - offset)
    else
        ts = start
        te = start + offset
    end

    return traj_idx, egoid, ts, te
end

#=
Description
    This function samples n values from the set s without replacement, and 
    does not work with anything except a set s. Could use statsbase, but want 
    to avoid the dependency.

Args:
    - s: a set
    - n: number of values to sample

Returns:
    - a subset of the values in s, as a list containing n elements
=#
function random_sample_from_set_without_replacement(s, n)
    @assert length(s) >= n
    sampled = Set()
    for i in 1:n
        cur = rand(s)
        push!(sampled, cur)
        delete!(s, cur)
    end
    return collect(sampled)
end

function sample_multiple_trajdata_vehicle(
        n_veh::Int,
        trajinfos, 
        offset::Int;
        max_resamples::Int = 100,
        egoid::Union{Void, Int} = nothing,
        traj_idx::Union{Void, Int} = nothing,
        verbose::Bool = true,
        rseed::Union{Void, Int} = nothing)
    
    if rseed != nothing
        srand(rseed)
    end
    # if passed in egoid and traj_idx, use those, otherwise, sample
    if egoid == nothing || traj_idx == nothing 
        # sample the ngsim trajectory
        traj_idx = rand(1:length(trajinfos))
        # sample the first vehicle and start and end timesteps
        egoid = rand(collect(keys(trajinfos[traj_idx])))
    end

    ts = trajinfos[traj_idx][egoid]["ts"]
    te = trajinfos[traj_idx][egoid]["te"]
    # this sampling assumes ts:te-offset is a valid range
    # this is enforced by the initial computation of the index / trajinfo
    ts = rand(ts:te - offset)
    # after setting the start timestep randomly from the valid range, next 
    # update the end timestep to be offset timesteps following it 
    # this assume that we just want to simulate for offset timesteps
    te = ts + offset

    # find all other vehicles that have at least 'offset' many steps in common 
    # with the first sampled egoid starting from ts. If the number of such 
    # vehicles is fewer than n_veh, then resample
    # start with the set containing the first egoid so we don't double count it
    egoids = Set{Int}(egoid)
    for othid in keys(trajinfos[traj_idx])
        oth_ts = trajinfos[traj_idx][othid]["ts"]
        oth_te = trajinfos[traj_idx][othid]["te"]
        # other vehicle must start at or before ts and must end at or after te
        if oth_ts <= ts && te <= oth_te
            push!(egoids, othid)
        end
    end

    # check that there are enough valid ids from which to sample
    if length(egoids) < n_veh
        # if not, resample
        # this is not ideal, but dramatically simplifies the multiagent env
        # if it becomes a problem, implement a version of the multiagent env 
        # with asynchronous resets
        if verbose
            println("WARNING: insuffcient sampling ids in sample_multiple_trajdata_vehicle, resamples remaining: $(max_resamples)")
        end
        if max_resamples == 0
            error("ERROR: reached maximum resamples in sample_multiple_trajdata_vehicle")
        else
            return sample_multiple_trajdata_vehicle(
                n_veh, 
                trajinfos, 
                offset, 
                max_resamples=max_resamples - 1,
                verbose=verbose)
        end
    end

    # reaching this point means there are sufficient ids, sample the ones to use
    egoids = random_sample_from_set_without_replacement(egoids, n_veh)

    return traj_idx, egoids, ts, te
end

function build_feature_extractor(params = Dict())
    subexts::Vector{AbstractFeatureExtractor} = []
    push!(subexts, CoreFeatureExtractor())
    push!(subexts, TemporalFeatureExtractor())
    push!(subexts, WellBehavedFeatureExtractor())
    push!(subexts, CarLidarFeatureExtractor(20, carlidar_max_range = 50.))
    push!(subexts, ForeForeFeatureExtractor())
    ext = MultiFeatureExtractor(subexts)
    return ext
end

function max_n_objects(trajdatas)
    cur_max = -1
    for trajdata in trajdatas
        cur = maximum(n_objects_in_frame(trajdata, i) for i in 1 : nframes(trajdata))
        cur_max = max(cur, cur_max)
    end
    return cur_max
end

function fill_infos_cache(ext::MultiFeatureExtractor)
    cache = Dict()
    cache["feature_names"] = feature_names(ext)
    for (i,n) in enumerate(cache["feature_names"])
        if "is_colliding" == n
            cache["is_colliding_idx"] = i
        end
        if "out_of_lane" == n 
            cache["out_of_lane_idx"] = i
        end
        if "markerdist_left" == n
            cache["markerdist_left_idx"] = i
        end
        if "markerdist_right" == n
            cache["markerdist_right_idx"] = i
        end
        if "accel" == n
            cache["accel_idx"] = i
        end
        if "distance_road_edge_right" == n
            cache["distance_road_edge_right_idx"] = i
        end
        if "distance_road_edge_left" == n
            cache["distance_road_edge_left_idx"] = i
        end

    end
    return cache
end

function keep_vehicle_subset!(scene::Scene, ids::Vector{Int})
    keep_ids = Set(ids)
    scene_ids = Set([veh.id for veh in scene])
    remove_ids = setdiff(scene_ids, keep_ids)
    for id in remove_ids
        delete!(scene, id)
    end
    return scene
end

function stack_tensor_dict_list(lst::Vector{Dict})
    dict_keys = collect(keys(lst[1]))
    ret = Dict()
    for k in dict_keys
        example = lst[1][k]
        if isa(example, Dict)
            v = stack_tensor_dict_list([x[k] for x in lst])
        else
            v = [x[k] for x in lst]
        end
        ret[k] = v
    end
    return ret
end
