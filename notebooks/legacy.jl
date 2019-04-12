# 5 car adjacent lanes scenario with 4 parameters
# Uses new functions as compared to 2 parameters case to enable working with more parameters
# These are: initialize_carwise_particle_buckets which, in turn, calls initialize_particles

# We see that even without specifying a ground truth value for v_des, the default value of 
# 29.0 is uncovered successfully. Same can't be said for the spacing and timegap params

num_p = 100 # number of particles
car_pos = [0.,0.,0.,0.,0.]

n_cars = length(car_pos) # number of cars
scene,roadway = init_scene_roadway(car_pos)

# Specify the ground truth values for v_des
# d1 = Dict(:v_des=>10.0,:σ=>0.2);d2 = Dict(:v_des=>20.0,:σ=>0.3);d3 = Dict(:v_des=>15.0,:σ=>0.)
# d4 = Dict(:v_des=>18.0,:σ=>0.4);d5 = Dict(:v_des=>27.0,:σ=>0.2)

# Test the default v_des uncovering
d1 = Dict(:σ=>0.2);d2 = Dict(:σ=>0.3);d3 = Dict(:σ=>0.)
d4 = Dict(:σ=>0.4);d5 = Dict(:σ=>0.2)

car_particles = [d1,d2,d3,d4,d5]

car_vel_array = [10.,20.,15.,20.,20.]

rec = generate_ground_truth(car_pos,car_particles,car_vel_array=car_vel_array,n_steps=100)

# loop over the trajectory step by step
f_end_num = length(rec.frames)

input = [(:T,0.1,0.1,10.),(:v_des,10.,0.1,30.),(:σ,0.1,0.1,1.),(:s_min,0.1,0.1,5.)]
bucket_array = initialize_carwise_particle_buckets(n_cars,num_p,input)
    
for t in 1:f_end_num-1
    if t%10==0 @show t end
    f = rec.frames[f_end_num - t + 1]
    
    for car_id in 1:n_cars
        old_p_set_dict = bucket_array[car_id]
        trupos = rec.frames[f_end_num-t].entities[car_id].state.posF.s
#         @show trupos
#         @show old_p_set_dict
        new_p_set_dict = update_p_one_step(roadway,f,trupos,old_p_set_dict,
            car_id=car_id,approach="pf")
        bucket_array[car_id] = new_p_set_dict
    end
end  
#@show fit(MvNormal,old_p_mat) # Don't work because all elements identical
print_buckets_mean(bucket_array)

##-------------------------------------------------------------------------------

# 5 car adjacent lane scenario with 2 parameters

num_p = 100 # number of particles
car_pos = [0.,0.,0.,0.,0.]

n_cars = length(car_pos) # number of cars
scene,roadway = init_scene_roadway(car_pos)
d1 = Dict(:v_des=>10.0,:σ=>0.2);d2 = Dict(:v_des=>20.0,:σ=>0.3);d3 = Dict(:v_des=>15.0,:σ=>0.)
d4 = Dict(:v_des=>18.0,:σ=>0.4);d5 = Dict(:v_des=>27.0,:σ=>0.2)

car_particles = [d1,d2,d3,d4,d5]

car_vel_array = [10.,20.,15.,20.,20.]

rec = generate_ground_truth(car_pos,car_particles,car_vel_array=car_vel_array,n_steps=100)

# loop over the trajectory step by step
f_end_num = length(rec.frames)

bucket_array = init_car_particle_buckets(n_cars,num_p)
    
for t in 1:f_end_num-1
    if t%10==0 @show t end
    f = rec.frames[f_end_num - t + 1]
    
    for car_id in 1:n_cars
        old_p_set_dict = bucket_array[car_id]
        trupos = rec.frames[f_end_num-t].entities[car_id].state.posF.s
#         @show trupos
#         @show old_p_set_dict
        new_p_set_dict = update_p_one_step(roadway,f,trupos,old_p_set_dict,
            car_id=car_id,approach="cem")
        bucket_array[car_id] = new_p_set_dict
    end
end  
#@show fit(MvNormal,old_p_mat) # Don't work because all elements identical
print_buckets_mean(bucket_array)
