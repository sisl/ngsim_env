"""
- Written on: Oct 15
- Purpose: Test the probability calculation of lane change
"""

include("utils.jl")

const TIMESTEP = 0.1;
const V_DES = 1; const SIGMA_IDM = 2; const T_HEADWAY = 3; const S_MIN=4; 
const POLITENESS = 5;const ADV_TH = 6;const SENSOR_SIGMA = 7

default_particle = [20.,0.,1.5,5.,0.35,0.1,NaN] # The parameters set to default values

pos_vel_array_1 = [(200.,30.),(250.,0.)]
pos_vel_array_2 = [(250.,10.)] #(280.,10.)
pos_vel_array_3 = [(215.,0.),(225.,10.),(230.,0.)]
lane_place_array = [pos_vel_array_1,pos_vel_array_2]
scene,roadway = init_place_cars(lane_place_array)

const ROADWAY = roadway

particle = [29.0,0.5,1.2,1.5,0.,0.1,20.]
start_step = 1; nsteps = 100
scene_halluc = deepcopy(scene)
models = Dict{Int64,DriverModel}()
for veh in scene
#     print("veh.id = $(veh.id)\n")
#     models[veh.id] = LatLonSeparableDriver(ProportionalLaneTracker(),IntelligentDriverModel())
    models[veh.id] = uncertain_IDM(sigma_sensor=20.)
end
models[1] = Tim2DDriver(TIMESTEP,
			mlane=MOBIL(TIMESTEP,politeness=particle[POLITENESS],mlon=uncertain_IDM(sigma_sensor=particle[SENSOR_SIGMA])
			),
			mlon = uncertain_IDM(sigma_sensor=particle[SENSOR_SIGMA]
			),
	    )
models[2] = uncertain_IDM(v_des=15.,sigma_sensor=particle[SENSOR_SIGMA])


halluc_scenes_list = []
lc_probs = fill(0.,nsteps,)
id = 1
for (i,t) in enumerate(start_step:start_step+nsteps-1)
    print("test_laneprob.jl says: t=$t\n")
    lp = get_lane_change_prob(scene_halluc,particle,car_id=id)
    #print("test_laneprob.jl says lane change prob = $(lp)\n")
    actions = Array{Any}(undef,length(scene_halluc))
    get_actions!(actions,scene_halluc,ROADWAY,models)
    tick!(scene_halluc,ROADWAY,actions,TIMESTEP)
    push!(halluc_scenes_list,deepcopy(scene_halluc))
    lc_probs[i] = lp
end

plot_ytrace = scenelist2ytrace(halluc_scenes_list,car_id=id)
plot_probs = PGFPlots.Plots.Scatter(collect(1:nsteps),lc_probs,legendentry="lane change probs")
PGFPlots.save("media/mobil/lc_ytrace.pdf",PGFPlots.Axis([plot_ytrace,plot_probs],
                                 xlabel="timestep",ylabel="ytrace and lc probs",
                                 legendPos="outer north east",
                                 title="Lane change prob and y pos trace for car id=$(id)"))
