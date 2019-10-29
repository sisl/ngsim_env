"""
- Written on: Oct 10
- Purpose: Make video showing headway sensor noise measurement causes differing lane change

- Oct 29: Debug why second lane change is so variable even with the same parameter
"""

include("utils.jl")


# GLOBAL DEFINITIONS: TIMESTEP, ROADWAY, PARTICLE INDICES
const TIMESTEP = 0.1;
const V_DES = 1; const SIGMA_IDM = 2; const T_HEADWAY = 3; const S_MIN=4; 
const POLITENESS = 5;const ADV_TH = 6;const SENSOR_SIGMA = 7;
const DEFAULT_PARTICLE = [20.,0.,1.5,5.,0.35,0.1,NaN] # The parameters set to default values

pos_vel_array_1 = [(200.,30.),(250.,0.)]
pos_vel_array_2 = [(250.,10.)] #(280.,10.)
pos_vel_array_3 = [(215.,0.),(225.,10.),(230.,0.)]
lane_place_array = [pos_vel_array_1,pos_vel_array_2]
scene,roadway = init_place_cars(lane_place_array)
const SCENE = deepcopy(scene)
const ROADWAY = roadway;

param_names = Dict(1=>"Desired velocity",2=>"Acceleration output noise",3=>"Min time headway",
    4=>"Min separation",5=>"Politeness",6=>"Advantage threshold",7=>"Headway sensor noise");

# GROUND TRUTH DATA: Default model params 100 timesteps, returns true_scene_list
true_scene_list = JLD.load("media/mobil/data_jld/ground_truth.jld","true_scene_list")

# EXPERIMENT: Do filtering and then generate imitation trajectories with different random seeds
#final_p_mat,iterwise_p_mat = multistep_update(car_id=1,start_frame=2,last_frame=99);

final_p_mat = JLD.load("media/mobil/data_jld/final_p.jld","final_p_mat")
seeds = collect(1:10)
car_id = 1
y_trace_plot = PGFPlots.Plots.Plot[]
lanes_plot = PGFPlots.Plots.Plot[]
for seed in seeds
    print("seed = $(seed)\n")
    start_scene = deepcopy(SCENE)
    Random.seed!(seed)
    scene_list = gen_imitation_traj(final_p_mat,start_scene,start_step=1,nsteps=100,car_id=1)
    push!(y_trace_plot,PGFPlots.Plots.Scatter(collect(1:length(scene_list)),
                [scene[car_id].state.posG.y for scene in scene_list],legendentry="seed = $seed"))
    #multiple_scenelist2video(true_scene_list,scene_list,filename = "media/mobil/true_vs_imitation_seed_$(seed)_debug.mp4")
    
end
pa = PGFPlots.Axis(y_trace_plot,xlabel="timestep",ylabel="y pos trace",
    title="Mean filtered particle y pos trace",legendPos="outer north east")
PGFPlots.save("media/mobil/debug_ytrace.pdf",pa)
