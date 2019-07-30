"""
Helper function for tests: init_scene: 
Generate a scene and roadway
Place car lanewise starting with lane 1 (rightmost)
Used for the 5 car side by side scenario. Cant be used for cars in the same lane 
i.e. interacting cars scenario
-----Arguments
`car_pos_array` Array with initial position of each car. Car i will be placed in lane i at that position
`car_vel_array` Array with initial velocities. Defaults to 0 velocities
-----Returns: `scene`, `roadway`
"""
function init_scene_roadway(car_pos_array;car_vel_array=[],road_length=1000.0,num_lanes = 5)
    roadway = gen_straight_roadway(num_lanes,road_length)
    n_cars = length(car_pos_array)
    
    if isempty(car_vel_array) car_vel_array=zeros(n_cars) end
    
    scene = Scene()
    for i in 1:n_cars
        veh_state = VehicleState(Frenet(roadway[LaneTag(1,i)], car_pos_array[i]), roadway, car_vel_array[i])
        veh = Vehicle(veh_state, VehicleDef(), i)
        push!(scene,veh)
    end

    return scene, roadway
end

"""
init_place_cars: Initializes a scene more generic than init_scene_roadway
init_scene_roadway would go element by element for input array and place each
car in a new lane. Was not possible to use that to place cars in the same lane

--------Arguments:
lane_place_array: Every element corresponds to a new lane starting
from lane1. Every element is an array. The number of elements in this
array is the number of cars in the same lane. Every element is a tuple.
Each tuple contains pos,vel for the car
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

"""
Generate a record with ground truth trajectory. Every frame is a scene containing 
ground truth positions of vehicles
-----------Arguments:
`car_pos_array` Array with initial position of each car. Car i will be placed in lane i at that position
`car_particle_array` Array with each element being a dictionary with true particle corresponding
to car_id equivalent to that index
`car_vel_array` Array with initial velocities. Defaults to 0 velocities
-----------Other functions called:`init_scene_roadway`

----------Returns: SceneRecord which is an array of frames all trough the trajectory
"""
function generate_ground_truth(car_pos_array,car_particle_array;car_vel_array = [],n_steps=100,dt=0.1)
    @assert length(car_pos_array) == length(car_particle_array)
    scene, roadway = init_scene_roadway(car_pos_array,car_vel_array=car_vel_array)
    n_cars = length(car_particle_array)
    models = Dict{Int, DriverModel}()
    
    # Create driver models for all the cars in the scene
    for i in 1:n_cars
        models[i] = IntelligentDriverModel(;car_particle_array[i]...)
    end
    
    rec = SceneRecord(n_steps, dt)
    simulate!(rec, scene, roadway, models, n_steps)
    return rec
end

"""
generate_truth_data: Ground truth generation function to work with init_place_array

- generate_ground_truth would work with init_scene_roadway which was not as generic as
the init_place_array because it allows placing multiple cars in the same lane

--------Arguments:
`lane_place_array`: Is the lane_place_array that is used with the init_place_cars
function
`car_particle_array`: Is the partiucles array that would need to make the

--------Other function called: `init_place_cars`

------Returns: scenerecord which is an array that contains the frames through the traj
"""
function generate_truth_data(lane_place_array,car_particle_array,n_steps=100,dt=0.1)
    scene,roadway = init_place_cars(lane_place_array)
    n_cars = length(car_particle_array)
    models = Dict{Int, DriverModel}()
    
    # Create driver models for all the cars in the scene
    for i in 1:n_cars
        models[i] = IntelligentDriverModel(;car_particle_array[i]...)
    end
    n_steps = 100;dt=0.1
    rec = SceneRecord(n_steps, dt)
    simulate!(rec, scene, roadway, models, n_steps)
    return rec
end

"""
Hallucinate a step forward given a specific car
---------Arguments:
`scene` Scene to start hallucination from
`particle` Dict with key as IDM parameter name and value as param val
`car_id` Identity of the car of interest

-------Returns: Hallucinated position of car of interest

---------NOTES:------------
- For now, we hallucinate the car of interest with the particle. Other cars are assumed to be
driving with IDM(v_des = 10). They don't matter in the particle update of the car of interest
(FOR NOW)
- We return only the position of the car along the lane. We don't even return which lane it is
in information. In the future, we want to return 2D position and measure 2D likelihood value somehow

# Example
td1 = load_trajdata(1);
scene = Scene(500)
get!(scene,td1,300)
display(scene.entities[2].state.posF.s)
roadway = ROADWAY_101;
particle = Dict(:v_des=>25.0,:σ=>0.5)
hallucinate_a_step(roadway,scene,particle,car_id=scene.entities[2].id)
"""
function hallucinate_a_step(roadway,scene_input,particle;car_id=-1)
    if car_id==-1 @show "Please give valid car_id" end
    
    scene = deepcopy(scene_input)
    #scene = scene_input # This was the failure case
    n_cars = scene.n 

    models = Dict{Int, DriverModel}()
    
    # Create driver models for all the cars in the scene
    for veh in scene
        if veh.id == car_id
            models[veh.id] = IntelligentDriverModel(;particle...)
        else
            # TODO: RESEARCH QUESTION: What drives the other vehicles in the hallucination
            models[veh.id] = IntelligentDriverModel(v_des=10.0)
        end
    end
    
    n_steps = 1
    dt = 0.1
    rec = SceneRecord(n_steps, dt)
    
    simulate!(rec, scene, roadway, models, n_steps)
    
    X = Array{Float64}(undef,n_steps, 1)

    for t in 1:n_steps
        f = rec.frames[n_steps - t + 1]
        
            # Access the vehicle with id as car_id and return its frenet s
        X[t,1] = scene.entities[findfirst(car_id,f)].state.posF.s

            # The above one liner in for loop fashion
#         for veh in f
#             if veh.id == car_id
#                 s = veh.state.posF
#                 X[t, 1] = s.s #position
#                 break
#             end
#         end
    end
    return X[1]
end
