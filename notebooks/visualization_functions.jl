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
function compare_trajectories(rec_true,rec_sim)
    @assert nframes(rec_sim) == nframes(rec_true)
    @manipulate for frame_index in 1:nframes(rec_sim)
        ghost_overlay = veh_overlay(rec_true[frame_index-nframes(rec_sim)])
        render(rec_sim[frame_index-nframes(rec_sim)], roadway, [ghost_overlay], canvas_height=100)
    end
end
# TODO: How do you test the compare_trajectories function?
