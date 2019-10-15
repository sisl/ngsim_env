using NGSIM
using AutomotiveDrivingModels
using AutoViz
using Reel
using Distributions
using PGFPlots
using JLD
using Random

include("utils.jl")

#----------------------------------------
#	Running script begins here
#----------------------------------------
# Define some global variables
const TIMESTEP = 0.1;
const V_DES = 1; const SIGMA_IDM = 2; const T_HEADWAY = 3; const S_MIN=4; const POLITENESS = 5;const ADV_TH = 6;const SENSOR_SIGMA = 7

pos_vel_array_1 = [(200.,30.),(250.,0.)]
pos_vel_array_2 = [(250.,10.)] #(280.,10.)
pos_vel_array_3 = [(215.,0.),(225.,10.),(230.,0.)]
lane_place_array = [pos_vel_array_1,pos_vel_array_2]
scene,roadway = init_place_cars(lane_place_array)

const ROADWAY = roadway


Random.seed!(5)
models = Dict{Int64,DriverModel}()
for veh in scene
#     print("veh.id = $(veh.id)\n")
#     models[veh.id] = LatLonSeparableDriver(ProportionalLaneTracker(),IntelligentDriverModel())
    models[veh.id] = uncertain_IDM(sigma_sensor=20.)
end
models[1] = Tim2DDriver(TIMESTEP,
			mlane=MOBIL(TIMESTEP,politeness=0.,mlon=uncertain_IDM(sigma_sensor=20.)
			),
			mlon = uncertain_IDM(sigma_sensor=20.
			),
	    )
models[2] = uncertain_IDM(v_des=15.,sigma_sensor=20.)

horizon = 100
scene_list = get_hallucination_scenes(scene,nsteps=horizon,models=models)

#JLD.save("media/mobil/6.jld","scene_list",scene_list)


#p1 = PGFPlots.Plots.Scatter(collect(1:horizon),
#    [get_lane_id(scene,1) for scene in scene_list[1:horizon]],legendentry="veh 1 lane profile")
#py1 = PGFPlots.Plots.Scatter(collect(1:horizon),
#    [scene[1].state.posG.y for scene in scene_list[1:horizon]],legendentry = "y pos")
#pa = PGFPlots.Axis([p1,py1],xlabel="timestep",ylabel="Lane number/y position",legendPos="outer north east")

#PGFPlots.save("media/mobil/debug_lane_profile_high.pdf",pa)
scenelist2video(scene_list,filename="media/mobil/debug_lane_changing_5_uncertainIDM.mp4")
#scenelist2video_quantized(scene_list,roadway=roadway,filename="media/mobil/debug_lane_changing_quantized.mp4")
