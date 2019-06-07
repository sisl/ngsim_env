# TODO: How do you write tests for these visualization functions?
# Define the overlay used to overlay the ground truth in white
struct veh_overlay <: SceneOverlay
    sim_scene::Scene # The simulated scene
end

function AutoViz.render!(rendermodel::RenderModel,overlay::veh_overlay, scene::Scene, roadway::Roadway)
    AutoViz.render!(rendermodel,overlay.sim_scene,car_color = colorant"white")
    return rendermodel
end

"""
Visualize true trajectory `rec_true` in white and simulated trajectory `rec_sim` in color
on the same video to compare the driving behavior

# Other functions used
- `veh_overlay`: A struct and associated render function to overlay a scene on top of another

-------------------------------
# Example
pos_vel_array_1 = [(0.,10.)]
pos_vel_array_2 = [(0.,0.)]
pos_vel_array_3 = [(0.,0.),(10.,10.)]
lane_place_array = [pos_vel_array_1,pos_vel_array_2,pos_vel_array_3]
scene,roadway = init_place_cars(lane_place_array)
d1 = Dict(:v_des=>10.0,:σ=>0.);d2 = Dict(:v_des=>10.0,:σ=>0.)
d3 = Dict(:v_des=>10.0,:σ=>0.);d4 = Dict(:v_des=>10.0,:σ=>0.)
car_particle_array = [d1,d2,d3,d4]
rec = generate_truth_data(lane_place_array,car_particle_array);

# Generate the simulation
pos_vel_array_1 = [(0.0+10.,10.)]
pos_vel_array_2 = [(0.0+10.,0.)]
pos_vel_array_3 = [(0.0+10.,0.),(10.0+10.,10.)]
lane_place_array = [pos_vel_array_1,pos_vel_array_2,pos_vel_array_3]
scene,roadway = init_place_cars(lane_place_array)
d1 = Dict(:v_des=>10.0,:σ=>0.);d2 = Dict(:v_des=>10.0,:σ=>0.)
d3 = Dict(:v_des=>10.0,:σ=>0.);d4 = Dict(:v_des=>10.0,:σ=>0.)
car_particle_array = [d1,d2,d3,d4]

rec_sim = generate_truth_data(lane_place_array,car_particle_array)

compare_trajectories(rec,rec_sim)
"""
function compare_trajectories(rec_true,rec_sim,roadway)
    @assert nframes(rec_sim) == nframes(rec_true)
    @manipulate for frame_index in 1:nframes(rec_sim)
        ghost_overlay = veh_overlay(rec_true[frame_index-nframes(rec_sim)])
        render(rec_sim[frame_index-nframes(rec_sim)], roadway, [ghost_overlay], canvas_height=100)
    end
end

"""
    estimate_then_make_video(num_p::Int64,n_cars::Int64,lane_place_array::Array,
        car_particles::Array,particle_props::Array;approach="pf")

Estimate parameters of IDM, then use the mean of the particle bucket for each car to generate
simulated trajectory. Make a video showing this simulated trajectory with ground truth
overlayed in white

# Other notable functions used
- `find_mean_particles_carwise`
- `compare_trajectories`

# Example (written for jupyter notebook)
	# 2 cars in the same lane scenario with 3 parameters
	# Parameter wise result plotting with both cem and pf in same plot
	# Make a video in the notebook using the @manipulate
num_particles = 100
pos_vel_array_1 = [(30.,18.),(10.,12.)]
lane_place_array = [pos_vel_array_1]
num_cars = 2
d1 = Dict(:v_des=>20.0,:σ=>0.1,:T=>1.5);d2 = Dict(:v_des=>10.0,:σ=>0.1,:T=>1.5)
car_particles = [d1,d2]
particle_props = [(:v_des,10.,0.1,30.),(:σ,0.1,0.1,1.),(:T,0.1,0.1,5.)]
estimate_then_make_video(num_particles,num_cars,lane_place_array,
    car_particles,particle_props,approach="pf")
"""
function estimate_then_make_video(num_p::Int64,n_cars::Int64,lane_place_array::Array,
        car_particles::Array,particle_props::Array;approach="pf")
    scene,roadway = init_place_cars(lane_place_array)
    rec = generate_truth_data(lane_place_array,car_particles)
    f_end_num = length(rec.frames)
    
    bucket_array = initialize_carwise_particle_buckets(n_cars,num_p,particle_props)
    for t in 1:f_end_num-1
        #if t%10==0 @show t end
        f = rec.frames[f_end_num - t + 1]

        for car_id in 1:n_cars
            old_p_set_dict = bucket_array[car_id]
            trupos = rec.frames[f_end_num-t].entities[car_id].state.posF.s
            new_p_set_dict = update_p_one_step(roadway,f,trupos,old_p_set_dict,
                car_id=car_id,approach=approach)
            bucket_array[car_id] = new_p_set_dict
        end
    end  
    
    # Estimation of parameters ends here. Now generate simulation
    car_particles_sim = find_mean_particle_carwise(bucket_array)
    sim_rec = generate_truth_data(lane_place_array,car_particles_sim)
    
    # Make a video with simulation in color and truth in white
    compare_trajectories(rec,sim_rec,roadway)
end

"""
	animate_record(rec_true::SceneRecord,rec_sim::SceneRecord,roadway,dt::Float64)

Function that is used by reel to generate video of driving behavior

# Functions used
- `veh_overlay` defined overlay to show ghost vehicles

# Example
```
duration, fps, render_hist = animate_record(rec, sim_rec, roadway, 0.1)
film = roll(render_hist, fps = fps, duration = duration)
```
"""
function animate_record(rec_true::SceneRecord,rec_sim::SceneRecord,
        roadway,dt::Float64)
    duration =rec_sim.nframes*dt
    fps = Int(1/dt)
    function render_rec(t, dt)
        frame_index = Int(floor(t/dt)) + 1
        ghost_overlay = veh_overlay(rec_true[frame_index-nframes(rec_sim)])
        return render(rec_sim[frame_index-nframes(rec_sim)], roadway,
            [ghost_overlay],canvas_height=100)
    end
    return duration, fps, render_rec
end

"""
    function viz_cem_pf_ghost(num_p::Int64,n_cars::Int64,lane_place_array::Array,
        car_particles::Array,particle_props::Array)
Function to generate simulations using parameters learned using both vanilla
and cem particle filtering
Returns resulting trajectories along with the ground truth trajectory

# Example
	# 2 car same lane scenario
num_particles = 100
pos_vel_array_1 = [(30.,18.),(10.,12.)]
lane_place_array = [pos_vel_array_1]
num_cars = 2
d1 = Dict(:v_des=>20.0,:σ=>0.1,:T=>1.5);d2 = Dict(:v_des=>10.0,:σ=>0.1,:T=>1.5)
car_particles = [d1,d2]
particle_props = [(:v_des,10.,0.1,30.),(:σ,0.1,0.1,1.),(:T,0.1,0.1,5.)]
rec,rec_sim_pf,rec_sim_cem,roadway = viz_cem_pf_ghost(num_particles,num_cars,lane_place_array,
    car_particles,particle_props)
"""
function viz_cem_pf_ghost(num_p::Int64,n_cars::Int64,lane_place_array::Array,
        car_particles::Array,particle_props::Array)
    scene,roadway = init_place_cars(lane_place_array)
    rec = generate_truth_data(lane_place_array,car_particles)
    f_end_num = length(rec.frames)
    
    # -------------vanilla particle filter--------------
    bucket_array = initialize_carwise_particle_buckets(n_cars,num_p,particle_props)
    for t in 1:f_end_num-1
        #if t%10==0 @show t end
        f = rec.frames[f_end_num - t + 1]

        for car_id in 1:n_cars
            old_p_set_dict = bucket_array[car_id]
            trupos = rec.frames[f_end_num-t].entities[car_id].state.posF.s
            new_p_set_dict = update_p_one_step(roadway,f,trupos,old_p_set_dict,
                car_id=car_id,approach="pf")
            bucket_array[car_id] = new_p_set_dict
        end
    end  
    
    # Estimation of parameters ends here. Now generate simulation
    car_particles_pf = find_mean_particle_carwise(bucket_array)
    sim_rec_pf = generate_truth_data(lane_place_array,car_particles_pf)
    
    # -------------CEM particle filter--------------
    bucket_array = initialize_carwise_particle_buckets(n_cars,num_p,particle_props)
    for t in 1:f_end_num-1
        #if t%10==0 @show t end
        f = rec.frames[f_end_num - t + 1]

        for car_id in 1:n_cars
            old_p_set_dict = bucket_array[car_id]
            trupos = rec.frames[f_end_num-t].entities[car_id].state.posF.s
            new_p_set_dict = update_p_one_step(roadway,f,trupos,old_p_set_dict,
                car_id=car_id,approach="cem")
            bucket_array[car_id] = new_p_set_dict
        end
    end  
    
    # Estimation of parameters ends here. Now generate simulation
    car_particles_cem = find_mean_particle_carwise(bucket_array)
    sim_rec_cem = generate_truth_data(lane_place_array,car_particles_cem)
    
    return rec,sim_rec_pf,sim_rec_cem,roadway
end

# Overlay that will be used to overlay ground truth and cem onto the vanilla traj
struct my_overlay <: SceneOverlay
    scene::Scene
    color # Needs to be of form colorant"Colorname"
end

"""
Render method for `my_overlay`. Helpful for making the color choice
"""
function AutoViz.render!(rendermodel::RenderModel,overlay::my_overlay, 
        scene::Scene, roadway::Roadway)
    AutoViz.render!(rendermodel,overlay.scene,car_color = overlay.color)
    return rendermodel
end

"""
Make a gif with vanilla pf in pink, ground truth in white and cem pf in blue

# Example
	# First run the example code in `viz_cem_pf_ghost`
duration, fps, render_hist = make_gif(rec, rec_sim_pf,rec_sim_cem, roadway, 0.1)
film = roll(render_hist,fps=fps,duration=duration)
write("5car_ghost+pf+cem.gif",film)
"""
function make_gif(rec_true::SceneRecord,rec_sim_pf::SceneRecord,rec_sim_cem::SceneRecord,
    roadway,dt::Float64)
    duration =rec_sim_pf.nframes*dt
    fps = Int(1/dt)
    function render_rec(t, dt)
        frame_index = Int(floor(t/dt)) + 1
        overlay_truth = my_overlay(rec_true[frame_index-nframes(rec_sim_pf)],colorant"white")
        overlay_cem = my_overlay(rec_sim_cem[frame_index-nframes(rec_sim_pf)],colorant"blue")
        render(rec_sim_pf[frame_index-nframes(rec_sim_pf)], roadway, 
            [overlay_truth,overlay_cem], canvas_height=100)
    end
    return duration,fps,render_rec
end
