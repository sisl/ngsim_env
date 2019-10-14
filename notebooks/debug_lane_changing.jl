using NGSIM
using AutomotiveDrivingModels
using AutoViz
using Reel
using Distributions
using PGFPlots
using JLD

# Define some global variables
TIMESTEP = 0.1; ROADWAY = roadway_ngsim
V_DES = 1; SIGMA_IDM = 2; T_HEADWAY = 3; S_MIN=4; POLITENESS = 5; ADV_TH = 6; SENSOR_SIGMA = 7

"""
    IDOverlay
Display the ID on top of each entity in a scene.
# Fields
- `color::Colorant`
- `font_size::Int64`
"""
mutable struct IDOverlay <: SceneOverlay
    color::Colorant
    font_size::Int
end

function AutoViz.render!(rendermodel::RenderModel, overlay::IDOverlay, scene::Scene, 
                            env::E) where E
    font_size = overlay.font_size
    for veh in scene
        add_instruction!(rendermodel, render_text, ("$(veh.id)", veh.state.posG.x, 
                        veh.state.posG.y, font_size, overlay.color), incameraframe=true)
    end
    return rendermodel
end

"""
Place the cars in starting position according to `lane_place_array`
"""
function init_place_cars(lane_place_array;road_length = 1000.0)
    num_lanes = length(lane_place_array)
    roadway = gen_straight_roadway(num_lanes,road_length)
    scene = Scene()

    id = 1
    for i in 1:num_lanes
        for j in 1:length(lane_place_array[i])
            veh_state = VehicleState(Frenet(roadway[LaneTag(1,i)],
                    lane_place_array[i][j][1]),roadway,
                lane_place_array[i][j][2])
            veh = Vehicle(veh_state,VehicleDef(),id)
            push!(scene,veh)
            id+=1
        end
    end
    return scene,roadway
end

function get_hallucination_scenes(start_scene;nsteps,models,start_step=1,
        id_list=[],verbosity = false)
        # Setting up
    scene_halluc = start_scene
    halluc_scenes_list = []
#     scene_halluc = get_scene(start_step,traj) # Frame to start hallucination from
#     push!(halluc_scenes_list,deepcopy(scene_halluc))
    
    for (i,t) in enumerate(start_step:start_step+nsteps-1)
        print("\nhallucination says: t = $t\n")
        
        actions = Array{Any}(undef,length(scene_halluc))

            # Propagation of scene forward
        get_actions!(actions,scene_halluc,ROADWAY,models)

        tick!(scene_halluc,ROADWAY,actions,TIMESTEP)
        
        push!(halluc_scenes_list,deepcopy(scene_halluc))
    end 
    return halluc_scenes_list
end

function scenelist2video(scene_list;
    filename = "media/mobil/scene_to_video.mp4")
    frames = Frames(MIME("image/png"),fps = 10)
    
    # Loop over list of scenes and convert to video
    for i in 1:length(scene_list)
        scene_visual = render(scene_list[i],ROADWAY,
        [IDOverlay(colorant"white",12),TextOverlay(text=["frame=$(i)"],font_size=12)],
#         cam=FitToContentCamera(0.),
        cam = CarFollowCamera(1)
        )
        push!(frames,scene_visual)
    end
    print("Making video filename: $(filename)\n")
    write(filename,frames)
    return nothing
end

"""
    function get_lane_id(scene,car_id)
# Examples
```julia
get_lane_id(scene,1)
```
"""
function get_lane_id(scene,car_id)
    veh = scene[findfirst(car_id,scene)]
    return veh.state.posF.roadind.tag.lane
end

function scenelist2video_quantized(scene_list;
    filename = "media/mobil/scene_to_video.mp4")
    frames = Frames(MIME("image/png"),fps = 5)
    
    # Loop over list of scenes and convert to video
    for i in 1:length(scene_list)
	if i%5 == 0
		scene_visual = render(scene_list[i],ROADWAY,
		[IDOverlay(colorant"white",12),TextOverlay(text=["frame=$(i)"],font_size=12)],
	#         cam=FitToContentCamera(0.),
		cam = CarFollowCamera(1)
		)
		push!(frames,scene_visual)
	end
    end
    print("Making video filename: $(filename)\n")
    write(filename,frames)
    return nothing
end

#-------------------Under development------------------------

function hallucinate_a_step(scene_input,particle;car_id=-1)
	if car_id==-1 @show "Please give a valid car_id" end
	
	scene = deepcopy(scene_input)
	models = Dict{Int64,DriverModel}()

	for veh in scene
		if veh.id == car_id
			models[veh.id] = Tim2DDriver(TIMESTEP,
						mlane=MOBIL(TIMESTEP,politeness=particle[POLITENESS],advantage_threshold=particle[ADV_TH]),
						mlon = IntelligentDriverModel(v_des=particle[V_DES],sigma=particle[SIGMA_IDM],
								T=particle[T_HEADWAY],s_min=particle[S_MIN])
						)
					)
					
		else
			models[veh.id] = IntelligentDriverModel(v_des=50.)
		end
	end
	actions = Array{Any}(undef,length(scene))
	get_actions!(actions,scene,ROADWAY,models)
	tick!(scene,ROADWAY,actions,TIMESTEP)

	halluc_state = scene.entities[findfirst(car_id,scene)].state
	halluc_pos = halluc_state.posF.s
	halluc_lane = get_lane_id(scene,car_id)

	return halluc_pos,halluc_lane
end

function compute_particle_likelihoods(roadway,f,trupos,p_set_dict;car_id=-1)
    if car_id==-1 @show "Please give valid car_id" end
    timestep = 0.1 #TODO: Remove hardcoding
    p_mat, params, vec_val_vec = to_matrix_form(p_set_dict)
    
    num_params=size(p_mat)[1]
    num_p = size(p_mat)[2]
    lkhd_vec = Array{Float64}(undef,num_p)
    for i in 1:num_p    
        # Create dict version for a single particle
        p_dict = Dict()
        for j in 1:num_params
            p_dict[params[j]]=vec_val_vec[j][i]
        end
        
        std_dev_acc = p_dict[:σ]
        
        # hack to avoid the std_dev_pos become negative and error Normal distb
        if std_dev_acc <= 0 std_dev_acc = 0.1 end
        
        # TODO: This math needs to be verified from random variable calculations
        std_dev_pos = timestep*timestep*std_dev_acc

        hpos = hallucinate_a_step(roadway,f,p_dict,car_id=car_id)
        lkhd_vec[i] = pdf(Normal(hpos,std_dev_pos),trupos[1])
    end
    return lkhd_vec,p_mat,params
end

"""
- Start hallucination from `start_scene` and compare resulting hallucination against ground truth at `trupos`, `trulane`
"""
function compute_particle_likelihoods(start_scene,true_nextpos,true_nextlane,p_mat;car_id=-1)
    if car_id==-1 @show "Please give valid car_id" end
    num_p = size(p_mat[2])
    lkhd_vec = Array{Float64}(undef,num_p)
    for i in 1:num_p
        particle = p_mat[:,i]
        std_dev_acc = p_mat[SIGMA_IDM]
        if std_dev_acc <= 0 std_dev_acc = 0.1 end
        std_dev_pos = TIMESTEP*TIMESTEP*std_dev_acc
        hpos,hlane = hallucinate_a_step(start_scene,particle,car_id=car_id)
        start_lane = get_lane_id(start_scene,car_id)
	lane_has_changed = false
	if start_lane != true_nextlane
		lane_has_changed = true
	end

	p_lanechange = get_lane_change_prob(start_scene,particle)

	prob_lane = 0.5 # Initialize to random
	if lane_has_changed
		prob_lane = p_lanechange
	else
		prob_lane = 1-p_lanechange
	end
	prob_pos = pdf(Normal(hpos,std_dev_pos),true_nextpos)
	lkhd_vec[i] = prob_lane*prob_pos
    end
end

"""
- Probability of lane changing start from `start_scene` and hallucinating using `particle` for `car_id` using `num_samplings` hallucinations
"""
function get_lane_change_prob(start_scene,particle;car_id=-1,num_samplings=10)
    start_lane = get_lane_id(start_scene,car_id)
    changed_count = 0; unchanged_count = 0
    for i in 1:num_samplings
        hpos,hlane = hallucinate_a_step(start_scene,particle,car_id=car_id)
        if hlane == start_lane
            unchanged_count += 1
	else
	    changed_count += 1
	end
    end
    return changed_count/num_samplings
end

#----------------------------------------
#	Running script begins here
#----------------------------------------
pos_vel_array_1 = [(200.,30.),(250.,0.)]
pos_vel_array_2 = [(250.,10.)] #(280.,10.)
pos_vel_array_3 = [(215.,0.),(225.,10.),(230.,0.)]
lane_place_array = [pos_vel_array_1,pos_vel_array_2]
scene,roadway = init_place_cars(lane_place_array)

models = Dict{Int64,DriverModel}()
for veh in scene
#     print("veh.id = $(veh.id)\n")
#     models[veh.id] = LatLonSeparableDriver(ProportionalLaneTracker(),IntelligentDriverModel())
    models[veh.id] = IntelligentDriverModel()
end
models[1] = Tim2DDriver(timestep_ngsim,mlane=MOBIL(timestep_ngsim,politeness=0.))
models[2] = IntelligentDriverModel(v_des=15.)

horizon = 100
scene_list = get_hallucination_scenes(scene,nsteps=horizon,models=models,roadway=roadway)

#JLD.save("media/mobil/6.jld","scene_list",scene_list)


#p1 = PGFPlots.Plots.Scatter(collect(1:horizon),
#    [get_lane_id(scene,1) for scene in scene_list[1:horizon]],legendentry="veh 1 lane profile")
#py1 = PGFPlots.Plots.Scatter(collect(1:horizon),
#    [scene[1].state.posG.y for scene in scene_list[1:horizon]],legendentry = "y pos")
#pa = PGFPlots.Axis([p1,py1],xlabel="timestep",ylabel="Lane number/y position",legendPos="outer north east")

#PGFPlots.save("media/mobil/debug_lane_profile_high.pdf",pa)
scenelist2video(scene_list,roadway=roadway,filename="media/mobil/debug_lane_changing_6.mp4")
#scenelist2video_quantized(scene_list,roadway=roadway,filename="media/mobil/debug_lane_changing_quantized.mp4")
