"""
Helper function for tests: init_scene: 
Generate a scene and roadway
Place car lanewise starting with lane 1 (rightmost)
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
"""
function hallucinate_a_step(roadway,scene,particle;car_id=-1)
    if car_id==-1 @show "Please give valid car_id" end
    n_cars = scene.n 

    models = Dict{Int, DriverModel}()
    
    # Create driver models for all the cars in the scene
    for i in 1:n_cars
        if i == car_id
            models[i] = IntelligentDriverModel(;particle...)
        else
            # TODO: RESEARCH QUESTION: What drives the other vehicles in the hallucination
            models[i] = IntelligentDriverModel(v_des=10.0)
        end
    end
    
    n_steps = 1
    dt = 0.1
    rec = SceneRecord(n_steps, dt)
    simulate!(rec, scene, roadway, models, n_steps)
    
    X = Array{Float64}(undef,n_steps, 1)

    for t in 1:n_steps
        f = rec.frames[n_steps - t + 1]
        
        for c in car_id:car_id
            s = f.entities[c].state.posF
            X[t, 1] = s.s #position
        end
    end
    return X[1]
end
