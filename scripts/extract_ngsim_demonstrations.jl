# Raunak testing what shows up in what git branch

using AutomotiveDrivingModels
using AutoRisk
using HDF5
using NGSIM

function build_feature_extractor()
    subexts = [
	LaneIDFeatureExtractor(),	# May 2018: Added by Raunak when lane id added feature
        CoreFeatureExtractor(),
        TemporalFeatureExtractor(),
        WellBehavedFeatureExtractor(),
        CarLidarFeatureExtractor(20, carlidar_max_range = 50.),
        ForeForeFeatureExtractor()
    ]
    ext = MultiFeatureExtractor(subexts)
    return ext
end

function extract_features(
        ext,
        trajdata, 
        roadway, 
        timestep_delta, 
        record_length, 
        offset, 
        prime,
        maxframes)
    n_features = length(ext)
    max_n_objects = maximum(n_objects_in_frame(trajdata, i) for i in 1 : nframes(trajdata))
    scene = Scene(max_n_objects)
    rec = SceneRecord(record_length, 0.1, max_n_objects)
    features = Dict{Int, Array{Float64}}()
    ctr = 0
    n_frames = nframes(trajdata)

    for frame in (offset - prime : offset - 1)
        # prime the rec
        AutomotiveDrivingModels.update!(rec, get!(scene, trajdata, frame))
    end

    veh_features = pull_features!(ext, rec, roadway, 1)

    for frame in offset : (n_frames - offset)
        ctr += 1
        if maxframes != nothing && ctr >= maxframes
            break
        end

        print("\rframe $(frame) / $(n_frames - offset)")
            
        # update the rec
        AutomotiveDrivingModels.update!(rec, get!(scene, trajdata, frame))

        # every timestep_delta step, extract features
        if frame % timestep_delta == 0

            for (vidx, veh) in enumerate(scene)
                # extract features
                veh_features = pull_features!(ext, rec, roadway, vidx)
                
                # add entry to features if vehicle not yet encountered
                if !in(veh.id, keys(features))
                    features[veh.id] = zeros(n_features, 0)
                end

                # stack onto existing features
                features[veh.id] = cat(2, features[veh.id], 
                    reshape(veh_features, (n_features, 1)))
            end
        end
    end
    return features
end

function write_features(features, output_filepath, ext)
    n_features = length(ext)

    # compute max length across samples
    maxlen = 0
    for (traj_idx, feature_dict) in features
        for (veh_id, veh_features) in feature_dict
            maxlen = max(maxlen, size(veh_features, 2))
        end
    end
    println("max length across samples: $(maxlen)")

    # write trajectory features
    h5file = h5open(output_filepath, "w")
    for (traj_idx, feature_dict) in features

        feature_array = zeros(n_features, maxlen, length(feature_dict))
        for (idx, (veh_id, veh_features)) in enumerate(feature_dict)
            feature_array[:, 1:size(veh_features, 2), idx] = reshape(veh_features, (n_features, size(veh_features, 2), 1))
        end
        h5file["$(traj_idx)"] = feature_array

    end

    # write feature names
    attrs(h5file)["feature_names"] = feature_names(ext)
    close(h5file)
end

function extract_ngsim_features(
        timestep_delta = 1, # timesteps between feature extractions
        record_length = 10, # number of frames for record to track in the past
        offset = 500, # from ends of the trajectories
        prime = 10,
        maxframes = nothing; # nothing for no max
        output_filename = "ngsim.h5",
        n_expert_files = 1) # number of time periods for which to extract.

    ext = build_feature_extractor()
    features = Dict{Int, Dict{Int, Array{Float64}}}()

    tic()
    # extract 
    for traj_idx in 1:n_expert_files

        # setup
        trajdata = load_trajdata(traj_idx)
        roadway = get_corresponding_roadway(traj_idx)
        features[traj_idx] = extract_features(
            ext, 
            trajdata, 
            roadway, 
            timestep_delta, 
            record_length, 
            offset, 
            prime,
            maxframes
        )
    end
    toc()

    output_filepath = joinpath("../data/trajectories/", output_filename)
    println("output filepath: $(output_filepath)")
    write_features(features, output_filepath, ext)

end

function extract_simple_features(
        filepath,
        output_filepath,
        timestep_delta = 1, # timesteps between feature extractions
        record_length = 20, # number of frames for record to track in the past
        offset = 3, # from ends of the trajectories
        prime = 2,
        maxframes = nothing) # nothing for no max

    ext = build_feature_extractor()
    features = Dict{Int, Dict{Int, Array{Float64}}}()

    tic()
    # setup
    trajdata = load_trajdata(filepath)
    roadway = get_corresponding_roadway(1)
    features[1] = extract_features(
        ext, 
        trajdata, 
        roadway, 
        timestep_delta, 
        record_length, 
        offset, 
        prime,
        maxframes
    )
    toc()
    write_features(features, output_filepath, ext)
end


# NGSIM
extract_ngsim_features(output_filename="ngsim_addLaneID.h5", n_expert_files=6)

# DEBUG
# trajdata_filepath = "/Users/wulfebw/.julia/v0.5/NGSIM/data/2_simple.txt"
# output_filepath = "../data/trajectories/2_simple.h5"
# extract_simple_features(trajdata_filepath, output_filepath)
