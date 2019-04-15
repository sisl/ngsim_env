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
