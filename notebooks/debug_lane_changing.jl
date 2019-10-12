using NGSIM
using AutomotiveDrivingModels
using AutoViz
using Reel
using Distributions
using PGFPlots
using JLD

timestep_ngsim = 0.1

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
        id_list=[],roadway,timestep=0.1,verbosity = false)
        # Setting up
    scene_halluc = start_scene
    halluc_scenes_list = []
#     scene_halluc = get_scene(start_step,traj) # Frame to start hallucination from
#     push!(halluc_scenes_list,deepcopy(scene_halluc))
    
    for (i,t) in enumerate(start_step:start_step+nsteps-1)
        print("\nhallucination says: t = $t\n")
        
        actions = Array{Any}(undef,length(scene_halluc))

            # Propagation of scene forward
        get_actions!(actions,scene_halluc,roadway,models)

        tick!(scene_halluc,roadway,actions,timestep)
        
        push!(halluc_scenes_list,deepcopy(scene_halluc))
    end 
    return halluc_scenes_list
end

function scenelist2video(scene_list;roadway,
    filename = "media/mobil/scene_to_video.mp4")
    frames = Frames(MIME("image/png"),fps = 10)
    
    # Loop over list of scenes and convert to video
    for i in 1:length(scene_list)
        scene_visual = render(scene_list[i],roadway,
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

function scenelist2video_quantized(scene_list;roadway,
    filename = "media/mobil/scene_to_video.mp4")
    frames = Frames(MIME("image/png"),fps = 5)
    
    # Loop over list of scenes and convert to video
    for i in 1:length(scene_list)
	if i%5 == 0
		scene_visual = render(scene_list[i],roadway,
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

function hallucinate_a_step(roadway,scene_input,particle;car_id=-1)
	if car_id==-1 @show "Please give a valid car_id" end
	
	scene = deepcopy(scene_input)
	models = Dict{Int64,DriverModel}()

	for veh in scene
		if veh.id == car_id
			models[veh.id] = IntelligentDriverModel()
		else
			models[veh.id] = IntelligentDriverModel(v_des=50.)
		end
	end
	actions = Array{Any}(undef,length(scene))
	get_actions!(actions,scene,roadway,models)
	tick!(scene,roadway,actions,timestep)

	halluc_state = scene.entities[findfirst(car_id,scene)].state
	halluc_pos = halluc_state.posF.s
	halluc_pos = get_lane_id(scene,car_id)
end

function overlay_jld_scenelists()
	scene_list_1 = JLD.load("media/mobil/1.jld", "scene_list")
	scene_list_2 = JLD.load("media/mobil/2.jld", "scene_list")
	scene_list_3 = JLD.load("media/mobil/3.jld", "scene_list")
	horizon = 100
	py1 = PGFPlots.Plots.Scatter(collect(1:horizon),
	    [scene[1].state.posG.y for scene in scene_list_1[1:horizon]],legendentry = "scenario 1")
	py2 = PGFPlots.Plots.Scatter(collect(1:horizon),
	    [scene[1].state.posG.y for scene in scene_list_2[1:horizon]],legendentry = "scenario 2")
	py3 = PGFPlots.Plots.Scatter(collect(1:horizon),
	    [scene[1].state.posG.y for scene in scene_list_3[1:horizon]],legendentry = "scenario 3")
	pa = PGFPlots.Axis([py1,py2,py3],xlabel="timestep",ylabel="y position")
	PGFPlots.save("media/mobil/overlay_scenes.pdf",pa)
	return nothing
end

#-------------------Running script begins here-----------------------
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
