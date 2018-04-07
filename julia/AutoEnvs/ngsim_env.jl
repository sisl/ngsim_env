export 
    NGSIMEnv,
    reset,
    step,
    observation_space_spec,
    action_space_spec,
    obs_names,
    render

#=
Description:
    NGSIM env that plays NGSIM trajectories, allowing the agent to take the place 
    of one of the vehicles in the trajectory
=#
type NGSIMEnv <: Env
    trajdatas::Vector{ListRecord}
    trajinfos::Vector{Dict}
    roadways::Vector{Roadway}
    roadway::Union{Void, Roadway} # current roadway
    scene::Scene
    rec::SceneRecord
    ext::MultiFeatureExtractor
    egoid::Int # current id of relevant ego vehicle
    ego_veh::Union{Void, Vehicle} # the ego vehicle
    traj_idx::Int # current index into trajdatas 
    t::Int # current timestep in the trajdata
    h::Int # current maximum horizon for egoid
    H::Int # maximum horizon
    primesteps::Int # timesteps to prime the scene
    Δt::Float64

    # settings
    terminate_on_collision::Bool
    terminate_on_off_road::Bool

    # metadata
    epid::Int # episode id
    render_params::Dict # rendering options
    infos_cache::Dict # cache for infos intermediate results
    function NGSIMEnv(
            params::Dict; 
            trajdatas::Union{Void, Vector{ListRecord}} = nothing,
            trajinfos::Union{Void, Vector{Dict}} = nothing,
            roadways::Union{Void, Vector{Roadway}} = nothing,
            reclength::Int = 5,
            Δt::Float64 = .1,
            primesteps::Int = 50,
            H::Int = 50,
            terminate_on_collision::Bool = true,
            terminate_on_off_road::Bool = true,
            render_params::Dict = Dict("zoom"=>5., "viz_dir"=>"/tmp"))
        param_keys = keys(params)
        @assert in("trajectory_filepaths", param_keys)

        # optionally overwrite defaults
        reclength = get(params, "reclength", reclength)
        primesteps = get(params, "primesteps", primesteps)
        H = get(params, "H", H)
        for (k,v) in get(params, "render_params", render_params)
            render_params[k] = v
        end
        terminate_on_collision = get(params, "terminate_on_collision", terminate_on_collision)
        terminate_on_off_road = get(params, "terminate_on_off_road", terminate_on_off_road)

        # load trajdatas if not provided
        if trajdatas == nothing || trajinfos == nothing || roadways == nothing
            trajdatas, trajinfos, roadways = load_ngsim_trajdatas(
                params["trajectory_filepaths"],
                minlength=primesteps + H
            )
        end

        # build components
        scene_length = max_n_objects(trajdatas)
        scene = Scene(scene_length)
        rec = SceneRecord(reclength, Δt, scene_length)
        ext = build_feature_extractor(params)
        infos_cache = fill_infos_cache(ext)
        return new(
            trajdatas, 
            trajinfos, 
            roadways,
            nothing,
            scene, 
            rec, 
            ext, 
            0, nothing, 0, 0, 0, H, primesteps, Δt,
            terminate_on_collision, terminate_on_off_road,
            0, render_params, infos_cache
        )
    end
end
function reset(
        env::NGSIMEnv; 
        offset::Int=env.H + env.primesteps,
        egoid::Union{Void,Int}=nothing, 
        start::Union{Void,Int}=nothing,
        traj_idx::Int=1)
    # sample the trajectory, ego vehicle
    env.traj_idx, env.egoid, env.t, env.h = sample_trajdata_vehicle(
        env.trajinfos, 
        offset,
        traj_idx,
        egoid,
        start
    )  

    # update / reset containers
    env.epid += 1
    empty!(env.rec)
    empty!(env.scene)
    
    # prime 
    for t in env.t:(env.t + env.primesteps)
        update!(env.rec, get!(env.scene, env.trajdatas[env.traj_idx], t))
    end
    # set the ego vehicle
    env.ego_veh = env.scene[findfirst(env.scene, env.egoid)]
    # set the roadway
    env.roadway = env.roadways[env.traj_idx]
    # env.t is the next timestep to load
    env.t += env.primesteps + 1
    # enforce a maximum horizon 
    env.h = min(env.h, env.t + env.H)
    return get_features(env)
end 

#=
Description:
    Propagate a single vehicle through an otherwise predeterined trajdata

Args:
    - env: environment to be stepped forward
    - action: array of floats that can be converted into an AccelTurnrate
=#
function _step!(env::NGSIMEnv, action::Array{Float64})
    # convert action into form 
    ego_action = AccelTurnrate(action...)
    # propagate the ego vehicle 
    ego_state = propagate(
        env.ego_veh, 
        ego_action, 
        env.roadway, 
        env.Δt
    )
    # update the ego_veh
    env.ego_veh = Entity(env.ego_veh, ego_state)

    # load the actual scene, and insert the vehicle into it
    get!(env.scene, env.trajdatas[env.traj_idx], env.t)
    vehidx = findfirst(env.scene, env.egoid)
    orig_veh = env.scene[vehidx] # for infos purposes
    env.scene[vehidx] = env.ego_veh

    # update rec with current scene 
    update!(env.rec, env.scene)

    # compute info about the step
    step_infos = Dict{String, Float64}()
    step_infos["rmse_pos"] = sqrt(abs2((orig_veh.state.posG - env.ego_veh.state.posG)))
    step_infos["rmse_vel"] = sqrt(abs2((orig_veh.state.v - env.ego_veh.state.v)))
    step_infos["rmse_t"] = sqrt(abs2((orig_veh.state.posF.t - env.ego_veh.state.posF.t)))
    step_infos["x"] = env.ego_veh.state.posG.x
    step_infos["y"] = env.ego_veh.state.posG.y
    step_infos["s"] = env.ego_veh.state.posF.s
    step_infos["phi"] = env.ego_veh.state.posF.ϕ
    return step_infos
end

function _extract_rewards(env::NGSIMEnv, infos::Dict{String, Float64})
    r = 0
    if infos["is_colliding"] == 1
        r -= 1
    end
    if infos["is_offroad"] == 1
        r -= 1
    end
    return r
end

function Base.step(env::NGSIMEnv, action::Array{Float64})
    step_infos = _step!(env, action)
    # compute features and feature_infos 
    features = get_features(env)
    feature_infos = _compute_feature_infos(env, features)
    # combine infos 
    infos = merge(step_infos, feature_infos)
    # update env timestep to be the next scene to load
    env.t += 1
    # compute terminal
    if env.t >= env.h
        terminal = true
    elseif env.terminate_on_collision && infos["is_colliding"] == 1
        terminal = true
    elseif env.terminate_on_off_road && (abs(infos["markerdist_left"]) > 3 && abs(infos["markerdist_right"]) > 3)
        terminal = true
    else
        terminal = false
    end
    reward = _extract_rewards(env, infos)
    return features, reward, terminal, infos
end
function _compute_feature_infos(env::NGSIMEnv, features::Array{Float64})
    is_colliding = features[env.infos_cache["is_colliding_idx"]]
    markerdist_left = features[env.infos_cache["markerdist_left_idx"]]
    markerdist_right = features[env.infos_cache["markerdist_right_idx"]]
    is_offroad = features[env.infos_cache["out_of_lane_idx"]]
    return Dict{String, Float64}(
        "is_colliding"=>is_colliding, 
        "markerdist_left"=>markerdist_left,
        "markerdist_right"=>markerdist_right,
        "is_offroad"=>is_offroad
    )
end
function AutoRisk.get_features(env::NGSIMEnv)
    veh_idx = findfirst(env.scene, env.egoid)
    pull_features!(env.ext, env.rec, env.roadway, veh_idx)
    return deepcopy(env.ext.features)
end
function observation_space_spec(env::NGSIMEnv)
    low = zeros(length(env.ext))
    high = zeros(length(env.ext))
    feature_infos = feature_info(env.ext)
    for (i, fn) in enumerate(feature_names(env.ext))
        low[i] = feature_infos[fn]["low"]
        high[i] = feature_infos[fn]["high"]
    end
    infos = Dict("high"=>high, "low"=>low)
    return (length(env.ext),), "Box", infos
end
action_space_spec(env::NGSIMEnv) = (2,), "Box", Dict("high"=>[4.,.15], "low"=>[-4.,-.15])
obs_names(env::NGSIMEnv) = feature_names(env.ext)

#=
Description:
    Render the scene 

Args:
    - env: environment to render

Returns:
    - img: returns a (height, width, channel) image to display
=#
function render(
        env::NGSIMEnv; 
        egocolor::Vector{Float64}=[1.,0.,0.],
        camtype::String="follow",
        static_camera_pos::Vector{Float64}=[0.,0.],
        camera_rotation::Float64=0.,
        canvas_height::Int=800,
        canvas_width::Int=800)
    # define colors for all the vehicles
    carcolors = Dict{Int,Colorant}()
    egocolor = ColorTypes.RGB(egocolor...)
    for veh in env.scene
        carcolors[veh.id] = veh.id == env.egoid ? egocolor : colorant"green"
    end

    # define a camera following the ego vehicle
    if camtype == "follow"
        cam = AutoViz.CarFollowCamera{Int}(env.egoid, env.render_params["zoom"])
    elseif camtype == "static"
        cam = AutoViz.StaticCamera(VecE2(static_camera_pos...), env.render_params["zoom"])
    else
        error("invalid camera type $(camtype)")
    end
    stats = [
        CarFollowingStatsOverlay(env.egoid, 2), 
        NeighborsOverlay(env.egoid, textparams = TextParams(x = 600, y_start=300))
    ]

    # rendermodel for optional rotation
    # note that for this to work, you have to comment out a line in AutoViz
    # src/overlays.jl:27 `clear_setup!(rendermodel)` in render
    rendermodel = RenderModel()
    camera_rotate!(rendermodel, deg2rad(camera_rotation))

    # render the frame
    frame = render(
        env.scene, 
        env.roadway,
        stats, 
        rendermodel = rendermodel,
        cam = cam, 
        car_colors = carcolors,
        canvas_height=canvas_height,
        canvas_width=canvas_width
    )

    # save the frame 
    if !isdir(env.render_params["viz_dir"])
        mkdir(env.render_params["viz_dir"])
    end
    ep_dir = joinpath(env.render_params["viz_dir"], "episode_$(env.epid)")
    if !isdir(ep_dir)
        mkdir(ep_dir)
    end
    filepath = joinpath(ep_dir, "step_$(env.t).png")
    write_to_png(frame, filepath)

    # load and return the frame as an rgb array
    img = PyPlot.imread(filepath)
    return img
end

