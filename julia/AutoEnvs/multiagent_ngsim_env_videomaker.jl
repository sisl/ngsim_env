export 
    MultiagentNGSIMEnvVideoMaker,
    reset,
    step,
    observation_space_spec,
    action_space_spec,
    obs_names,
    render

#=
Description:
    Multiagent NGSIM env that plays NGSIM trajectories, allowing a variable 
    number of agents to simultaneously control vehicles in the scene

    Raunak: This is basically a copy of multiagent_ngsim_env.jl with just 
    a few additions to enable color coded video making
=#
type MultiagentNGSIMEnvVideoMaker <: Env
    trajdatas::Vector{ListRecord}
    trajinfos::Vector{Dict}
    roadways::Vector{Roadway}
    roadway::Union{Void, Roadway} # current roadway
    scene::Scene
    rec::SceneRecord
    ext::MultiFeatureExtractor
    egoids::Vector{Int} # current ids of relevant ego vehicles
    ego_vehs::Vector{Union{Void, Vehicle}} # the ego vehicles
    traj_idx::Int # current index into trajdatas 
    t::Int # current timestep in the trajdata
    h::Int # current maximum horizon for egoid
    H::Int # maximum horizon
    primesteps::Int # timesteps to prime the scene
    Δt::Float64

    # multiagent type members
    n_veh::Int # number of simultaneous agents
    remove_ngsim_veh::Bool # whether to remove ngsim veh from all scenes
    features::Array{Float64}

    # metadata
    epid::Int # episode id
    render_params::Dict # rendering options
    infos_cache::Dict # cache for infos intermediate results
    function MultiagentNGSIMEnvVideoMaker(
            params::Dict; 
            trajdatas::Union{Void, Vector{ListRecord}} = nothing,
            trajinfos::Union{Void, Vector{Dict}} = nothing,
            roadways::Union{Void, Vector{Roadway}} = nothing,
            reclength::Int = 5,
            Δt::Float64 = .1,
            primesteps::Int = 50,
            H::Int = 50,
            n_veh::Int = 20,
            remove_ngsim_veh::Bool = false,
            render_params::Dict = Dict("zoom"=>5., "viz_dir"=>"/tmp"))
        param_keys = keys(params)
        @assert in("trajectory_filepaths", param_keys)

        # optionally overwrite defaults
        reclength = get(params, "reclength", reclength)
        primesteps = get(params, "primesteps", primesteps)
        H = get(params, "H", H)
        n_veh = get(params, "n_veh", n_veh)
        remove_ngsim_veh = get(params, "remove_ngsim_veh", remove_ngsim_veh)
        for (k,v) in get(params, "render_params", render_params)
            render_params[k] = v
        end

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
        # features are stored in row-major order because they will be transferred
        # to python; this is inefficient in julia, but don't change or it will 
        # break the python side of the interaction
        features = zeros(n_veh, length(ext))
        egoids = zeros(n_veh)
        ego_vehs = [nothing for _ in 1:n_veh]
        return new(
            trajdatas, 
            trajinfos, 
            roadways,
            nothing,
            scene, 
            rec, 
            ext, 
            egoids, ego_vehs, 0, 0, 0, H, primesteps, Δt, 
            n_veh, remove_ngsim_veh, features,
            0, render_params, infos_cache
        )
    end
end

#=
Description:
    Reset the environment. Note that this environment maintains the following 
    invariant attribute: at any given point, all vehicles currently being controlled
    will end their episode at the same time. This simplifies the rest of the code 
    by enforcing synchronized restarts, but it does somewhat limit the sets of 
    possible vehicles that can simultaneously interact. With a small enough minimum 
    horizon (H <= 250 = 25 seconds) and number of vehicle (n_veh <= 100)
    this should not be a problem. If you need to run with larger numbers then those 
    implement an environment with asynchronous resets.

Args:
    - env: env to reset 
    - dones: bool vector indicating which indices have reached a terminal state 
        these must be either all true or all false
=#
function reset(
        env::MultiagentNGSIMEnvVideoMaker,
        dones::Vector{Bool} = fill!(Vector{Bool}(env.n_veh), true); 
        offset::Int=env.H + env.primesteps,
        random_seed::Union{Void, Int} = nothing)
    # enforce environment invariant reset property 
    # (i.e., always either all true or all false)
    @assert (all(dones) || all(.!dones))
    # first == all at this point, so if first is false, skip the reset
    if !dones[1]
        return
    end

    # sample multiple ego vehicles
    # as stated above, these will all end at the same timestep
    env.traj_idx, env.egoids, env.t, env.h = sample_multiple_trajdata_vehicle(
        env.n_veh,
        env.trajinfos, 
        offset,
        rseed=random_seed
    )  

    # update / reset containers
    env.epid += 1
    empty!(env.rec)
    empty!(env.scene)
    
    # prime 
    for t in env.t:(env.t + env.primesteps)
        get!(env.scene, env.trajdatas[env.traj_idx], t)
        if env.remove_ngsim_veh
            keep_vehicle_subset!(env.scene, env.egoids)
        end
        update!(env.rec, env.scene)
    end

    # set the ego vehicle
    for (i, egoid) in enumerate(env.egoids)
        vehidx = findfirst(env.scene, egoid)
        env.ego_vehs[i] = env.scene[vehidx]
    end
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
function _step!(env::MultiagentNGSIMEnvVideoMaker, action::Array{Float64})
    # make sure number of actions passed in equals number of vehicles
    @assert size(action, 1) == env.n_veh
    ego_states = Vector{VehicleState}(env.n_veh)
    # propagate all the vehicles and get their new states
    for (i, ego_veh) in enumerate(env.ego_vehs)
        # convert action into form 
	ego_action = AccelTurnrate(action[i,:]...)
        # propagate the ego vehicle 
        ego_states[i] = propagate(
            ego_veh, 
            ego_action, 
            env.roadway, 
            env.Δt
        )
        # update the ego_veh
        env.ego_vehs[i] = Entity(ego_veh, ego_states[i])
    end

    # load the actual scene, and insert the vehicles into it
    get!(env.scene, env.trajdatas[env.traj_idx], env.t)
    if env.remove_ngsim_veh
        keep_vehicle_subset!(env.scene, env.egoids)
    end
    orig_vehs = Vector{Vehicle}(env.n_veh)

    for (i, egoid) in enumerate(env.egoids)

	vehidx = findfirst(env.scene, egoid)

        # track the original vehicle for validation / infos purposes
        orig_vehs[i] = env.scene[vehidx]

	# replace the original with the controlled vehicle

        env.scene[vehidx] = env.ego_vehs[i]

	# Raunak testing how to access lane id
	# Commented out for now as not relevant to reward augmentation
	# lane4print = env.ego_vehs[i].state.posF.roadind.tag.lane
	# println("Lane number = $lane4print")

	# Raunak's failed attempt at Ghost vehicle. The below treats both original
	# and policy driven as real cars and ends up showing the both to be colliding
	#push!(env.scene,env.ego_vehs[i])
    end

    # update rec with current scene 
    update!(env.rec, env.scene)

    # compute info about the step
#    step_infos = Dict{String, Vector{Float64}}(
#        "rmse_pos"=>Float64[],
#        "rmse_vel"=>Float64[],
#        "rmse_t"=>Float64[],
#        "x"=>Float64[],
#        "y"=>Float64[],
#        "s"=>Float64[],
#        "phi"=>Float64[],
#    )
#    for i in 1:env.n_veh
#        push!(step_infos["rmse_pos"], sqrt(abs2((orig_vehs[i].state.posG - env.ego_vehs[i].state.posG))))
#        push!(step_infos["rmse_vel"], sqrt(abs2((orig_vehs[i].state.v - env.ego_vehs[i].state.v))))
#        push!(step_infos["rmse_t"], sqrt(abs2((orig_vehs[i].state.posF.t - env.ego_vehs[i].state.posF.t))))
#        push!(step_infos["x"], env.ego_vehs[i].state.posG.x)
#        push!(step_infos["y"], env.ego_vehs[i].state.posG.y)
#        push!(step_infos["s"], env.ego_vehs[i].state.posF.s)
#        push!(step_infos["phi"], env.ego_vehs[i].state.posF.ϕ)
#    end

    # Raunak adds in original vehicle properties
    step_infos = Dict{String, Vector{Float64}}(
        "rmse_pos"=>Float64[],
        "rmse_vel"=>Float64[],
        "rmse_t"=>Float64[],
        "x"=>Float64[],
        "y"=>Float64[],
        "s"=>Float64[],
        "phi"=>Float64[],
	"orig_x"=>Float64[],
	"orig_y"=> Float64[],
	"orig_theta"=>Float64[],
	"orig_length"=>Float64[],
	"orig_width"=>Float64[]
    )
    for i in 1:env.n_veh
        push!(step_infos["rmse_pos"], sqrt(abs2((orig_vehs[i].state.posG - env.ego_vehs[i].state.posG))))
        push!(step_infos["rmse_vel"], sqrt(abs2((orig_vehs[i].state.v - env.ego_vehs[i].state.v))))
        push!(step_infos["rmse_t"], sqrt(abs2((orig_vehs[i].state.posF.t - env.ego_vehs[i].state.posF.t))))
        push!(step_infos["x"], env.ego_vehs[i].state.posG.x)
        push!(step_infos["y"], env.ego_vehs[i].state.posG.y)
        push!(step_infos["s"], env.ego_vehs[i].state.posF.s)
        push!(step_infos["phi"], env.ego_vehs[i].state.posF.ϕ)
	push!(step_infos["orig_x"], orig_vehs[i].state.posG.x)
	push!(step_infos["orig_y"], orig_vehs[i].state.posG.y)
	push!(step_infos["orig_theta"], orig_vehs[i].state.posG.θ)
	push!(step_infos["orig_length"], orig_vehs[i].def.length)
	push!(step_infos["orig_width"], orig_vehs[i].def.width)
    end

    return step_infos
end

function _extract_rewards(env::MultiagentNGSIMEnvVideoMaker, infos::Dict{String, Array{Float64}})
    rewards = zeros(env.n_veh)
    R = 0
    
    for i in 1:env.n_veh
        if infos["is_colliding"][i] == 1
            rewards[i] -= R
        elseif infos["is_offroad"][i] == 1
            rewards[i] -= R
        elseif infos["hard_brake"][i] == 1
            rewards[i] -= (R*0.5) # braking hard is not as bad as a collision
        end
    end
    return rewards
end

function Base.step(env::MultiagentNGSIMEnvVideoMaker, action::Array{Float64})
    step_infos = _step!(env, action)
    
    # compute features and feature_infos 
    features = get_features(env)
    feature_infos = _compute_feature_infos(env, features)
    
    # combine infos 
    infos = merge(step_infos, feature_infos)
    
    # update env timestep to be the next scene to load
    env.t += 1
    
    # compute terminal
    terminal = env.t > env.h ? true : false
    terminal = [terminal for _ in 1:env.n_veh]
    # vectorized sampler does not call reset on the environment
    # but expects the environment to handle resetting, so do that here
    # note: this mutates env.features in order to return the correct obs when resetting
    reset(env, terminal)
	rewards = _extract_rewards(env, infos)
    return deepcopy(env.features), rewards, terminal, infos
end

function _compute_feature_infos(env::MultiagentNGSIMEnvVideoMaker, features::Array{Float64};
                                                         accel_thresh::Float64=-3.0)
    feature_infos = Dict{String, Array{Float64}}(
                "is_colliding"=>Float64[], 
                "is_offroad"=>Float64[],
                "hard_brake"=>Float64[],
		"colliding_veh_ids"=>Float64[],
		"offroad_veh_ids"=>Float64[],
		"hardbrake_veh_ids"=>Float64[]
		)

    # Raunak explains: env.n_veh will be number of policy driven cars
    # Caution: This need not be the same as number of cars in the scene
    # Because the scene contains both policy driven cars and ngsim replay cars

    # Raunak debugging
    #sizeFeatures=size(features)
    #println("Compute repoting featurs are \n $sizeFeatures")
    #something = env.infos_cache
    #println("env.infos_cache =\n $something")
    #somethingelse = env.infos_cache["is_colliding_idx"]
    #println("env.infos_cache[is_colliding_idx]=$somethingelse")

    for i in 1:env.n_veh
	# Raunak debugging
	#println("i=$i\n")
	#egoVehid = env.ego_vehs[i].id 
	#println("egoVeh[i] = $egoVehid")
	#infos_cache = env.infos_cache
	#println("env.infos_cache = $infos_cache")
	
	is_colliding = features[i, env.infos_cache["is_colliding_idx"]]
	#println("is_colliding=$is_colliding\n")
	is_offroad = features[i, env.infos_cache["out_of_lane_idx"]]
        accel = features[i, env.infos_cache["accel_idx"]]
        push!(feature_infos["hard_brake"], accel <= accel_thresh)
        push!(feature_infos["is_colliding"], is_colliding)
        push!(feature_infos["is_offroad"], is_offroad)
	
	# Raunak adding list of colliding ego ids into the feature list that gets passed to render
	if is_colliding==1
		push!(feature_infos["colliding_veh_ids"],env.ego_vehs[i].id)
		println("Collision has happened see red")
	end
	if is_offroad==1
		push!(feature_infos["offroad_veh_ids"],env.ego_vehs[i].id)
		println("Offroad has happened see yellow")
	end
	if accel <= accel_thresh
		push!(feature_infos["hardbrake_veh_ids"],env.ego_vehs[i].id)
		println("Hard brake has happened see some color")
	end
    end
    return feature_infos
end


function AutoRisk.get_features(env::MultiagentNGSIMEnvVideoMaker)
    for (i, egoid) in enumerate(env.egoids)
	#println("i=$i\n")
	#println("egoid = $egoid\n")
	veh_idx = findfirst(env.scene, egoid)
	#println("veh_idx=$veh_idx")
	pull_features!(env.ext, env.rec, env.roadway, veh_idx)
        env.features[i, :] = deepcopy(env.ext.features)

	# Raunak checking whether lane_id got added as a feature
	# Does changing code in the julia folder automatically reflect
	#feature4print = env.ext.features
	#println("features are $feature4print")
    end
    return deepcopy(env.features)
end


function observation_space_spec(env::MultiagentNGSIMEnvVideoMaker)
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
action_space_spec(env::MultiagentNGSIMEnvVideoMaker) = (2,), "Box", Dict("high"=>[4.,.15], "low"=>[-4.,-.15])
obs_names(env::MultiagentNGSIMEnvVideoMaker) = feature_names(env.ext)
vectorized(env::MultiagentNGSIMEnvVideoMaker) = true
num_envs(env::MultiagentNGSIMEnvVideoMaker) = env.n_veh

# Raunak defined this render function to enable color changing based on collisions and offroads
function render(
        env::MultiagentNGSIMEnvVideoMaker;
	infos=Dict(),
        egocolor::Vector{Float64}=[0.,0.,1.],
        camtype::String="follow",
        static_camera_pos::Vector{Float64}=[0.,0.],
        camera_rotation::Float64=0.,
        canvas_height::Int=800,
        canvas_width::Int=800)
    # define colors for all the vehicles
    carcolors = Dict{Int,Colorant}()
    egocolor = ColorTypes.RGB(egocolor...)

    # Loop over all the vehicles in the scene. Note these may be both policy driven and ngsim replay
    for veh in env.scene
	# If current vehicle is a policy driven vehicle then color it blue otherwise color it green
	carcolors[veh.id] = in(veh.id, env.egoids) ? egocolor : colorant"green"
	
	# If the current vehicle is in the list of colliding vehicles color it red
	if in(veh.id,infos["colliding_veh_ids"])
		carcolors[veh.id] = ColorTypes.RGB([1.,0.,0.]...)
	end

	# If current vehicle is in the list of offroad vehicles color it yellow
	if in(veh.id,infos["offroad_veh_ids"])
		carcolors[veh.id]=ColorTypes.RGB([1.,1.,0.]...)
	end

	# If current vehicle is in the list of hard brakers then color it light blue
	if in(veh.id,infos["hardbrake_veh_ids"])
		carcolors[veh.id]=ColorTypes.RGB([0.,1.,1.]...)
	end
    end

    # define a camera following the ego vehicle
    if camtype == "follow"
        # follow the first vehicle in the scene
        cam = AutoViz.CarFollowCamera{Int}(env.egoids[1], env.render_params["zoom"])
    elseif camtype == "static"
        cam = AutoViz.StaticCamera(VecE2(static_camera_pos...), env.render_params["zoom"])
    else
        error("invalid camera type $(camtype)")
    end
    
    # Raunak commented this out because it was creating rays that were being used for
    # some research that Tim had been doing
    overlays = [
        CarFollowingStatsOverlay(env.egoids[1], 2), 
    #    NeighborsOverlay(env.egoids[1], textparams = TextParams(x = 600, y_start=300))
    ]

    # Raunak video plotting the ghost vehicle
    # See 'OrigVehicleOverlay' defined in AutoViz/src/2d/overlays.jl
    # to understand how the ghost vehicle is being plotted
#    overlays = [
#       CarFollowingStatsOverlay(env.egoids[1], 2)
#	,OrigVehicleOverlay(infos["orig_x"][1],infos["orig_y"][1],infos["orig_theta"][1],infos["orig_length"][1],infos["orig_width"][1])
#    ]


    # rendermodel for optional rotation
    # note that for this to work, you have to comment out a line in AutoViz
    # src/overlays.jl:27 `clear_setup!(rendermodel)` in render
    rendermodel = RenderModel()
    camera_rotate!(rendermodel, deg2rad(camera_rotation))

    # render the frame
    frame = render(
        env.scene, 
        env.roadway,
        overlays, 
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

# Raunak trying to add the original vehicle as a ghost car in the video
# Erroring out when defined here. Worked fine when defined within AutoViz/src/2d/Overlays.jl
# which is the same location where CarFollowingStatsOverlay is defined
#mutable struct OrigVehicleOverlay <: SceneOverlay
	# structure attributes here
#	x::Float64
#	y::Float64
#	theta::Float64
#	length::Float64
#	width::Float64
#	color::Colorant

	# constructor function

#	function OrigVehicleOverlay(x::Float64,y::Float64,theta::Float64,length::Float64,
#				    width::Float64,color::Colorant=colorant"white")
#	new(x,y,theta,length,width,color)
#	end
#end

# Better have this function signature exactly same as CarFollowingStatsOverlay
#function render!(rendermodel::RenderModel,overlay::OrigVehicleOverlay,scene::Scene, roadway::Roadway)
#	verbose = true
#	if verbose
#		println("render! within OrigVehicleOverlay reporting here")
#		x=overlay.x 
#		y=overlay.y
#		println("x=$x,y = $y")
#	end
#   add_instruction!(rendermodel, render_vehicle, (overlay.x, overlay.y, overlay.theta, 
#						   overlay.length, overlay.width, overlay.color))
#    return rendermodel
#end
