#-----------------------------------------------------------------------------
#			Imports
#-----------------------------------------------------------------------------

#%matplotlib inline

import h5py
from IPython.display import HTML
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import ndimage
import sys
import tensorflow as tf

import hgail.misc.utils

import hyperparams
import utils
import validate


#-----------------------------------------------------------------------------
#			Directory specifications and paths
#-----------------------------------------------------------------------------
basedir = '../../data/experiments/'
model_labels = [
#    'laneidtest_0_2_fine',
#    'continuous_normalized_laneid_0_2_fine',
#    'continuous_normalized_laneid_0_3_fine',
#    'continuous_laneid_0_2_fine',
    'discreteLaneID_RAIL_2000_2_fine',
]
itrs = [
    200,
#    200
]
model_params_filepaths = [os.path.join(basedir, label, 'imitate/log/itr_' + str(itrs[i]) + '.npz') 
                          for i,label in enumerate(model_labels)]
model_args_filepaths = [os.path.join(basedir, label, 'imitate/log/args.npz') for label in model_labels]
n_models = len(model_labels)

multi = True


#-----------------------------------------------------------------------------
#			FUNCTION: MULTIAGENT SIMULATE
#-----------------------------------------------------------------------------
def mutliagent_simulate(env, policy, max_steps, env_kwargs=dict(), render_kwargs=dict()):
    # Raunak: This x is the features arranged in shape (num_agents,num_features)
    x = env.reset(**env_kwargs) 
    n_agents = x.shape[0]
    traj = hgail.misc.simulation.Trajectory()
    dones = [True] * n_agents
    policy.reset(dones)
    imgs = []
    
    # Raunak adding for lane change rate calculation
    prevLaneNumsVector = np.zeros(n_agents) 
    numLaneChanges = 0
    
    
    for step in range(max_steps):
        sys.stdout.write('\rstep: {} / {}'.format(step+1, max_steps))
        a, a_info = policy.get_actions(x)
        
        #************************** Raunak tinkering
        #print(a[0][1])
        #a[0][0] = - 1.0  # Slows car down and then makes it drive in reverse
        #a[0][1] = - 1.0   # Turns car to the right
        #*************************************************
        nx, r, dones, e_info = env.step(a)
        traj.add(x, a, r, a_info, e_info)

        # Adding in the features as an argument to render 
        # to enable collision, offroad and ghost car
        render_kwargs['infos']=e_info
        
        # Raunak's version of render within multiagent_ngsim_env.jl that allows coloring
        img = env.render(**render_kwargs)  
        
        imgs.append(img)
        
        if any(dones): break
        x = nx

        #************************Raunak lane change rate testing************
        # Get the current lane id vector
        currentLaneNumsVector = e_info['laneNum']
        
        # Perform subtraction to calculate difference
        laneChangesVector = currentLaneNumsVector - prevLaneNumsVector
        
        numLaneChanges += np.count_nonzero(laneChangesVector)
        
        # Assign current to become previous lane id vector for next time step
        prevLaneNumsVector = currentLaneNumsVector

        # Verbose for finding lane change timesteps
        #if np.count_nonzero(laneChangesVector) != 0:
        #    print("Haha found a change\n")
        #    print("laneChangeVector = ",laneChangesVector)


    print("numLaneChanges = ",numLaneChanges)
    return imgs


#-----------------------------------------------------------------------------
#			FUNCTION: CREATE RENDER MAP
#-----------------------------------------------------------------------------
def create_render_map(model_labels, model_args_filepaths, model_params_filepaths, 
                      multi=False, rand=None, max_steps=200, n_vehs=None, remove_ngsim=False):
    render_map = dict()
    env_kwargs = dict()
    if rand != None:
        env_kwargs = dict(random_seed=rand)
    if not multi:
        env_kwargs = dict(
            egoid=worst_egoid, 
            start=worst_start
        )
    render_kwargs = dict(
        camera_rotation=45.,
        canvas_height=500,
        canvas_width=600
    )
    for i in range(len(model_labels)):
        print('\nrunning: {}'.format(model_labels[i]))

        # create session
        tf.reset_default_graph()
        sess = tf.InteractiveSession()

        # load args and params
        args = hyperparams.load_args(model_args_filepaths[i])
        print('\nargs loaded from {}'.format(model_args_filepaths[i]))
        if multi:
            args.env_multiagent = True
            if remove_ngsim:
                args.remove_ngsim_veh = True

            if n_vehs:
                args.n_envs = 100
                args.n_vehs = 100
        params = hgail.misc.utils.load_params(model_params_filepaths[i])
        print('\nparams loaded from {}'.format(model_params_filepaths[i]))
        
        # load env and params
        # Raunak adding in an argument for videmaking
        # See build_ngsim_env in utils.py for what this does
        env, _, _ = utils.build_ngsim_env(args,videoMaking=True)
        print("Raunak says: This is the videmaker reporting")
        normalized_env = hgail.misc.utils.extract_normalizing_env(env)
        if normalized_env is not None:
            normalized_env._obs_mean = params['normalzing']['obs_mean']
            normalized_env._obs_var = params['normalzing']['obs_var']

        # load policy
        if 'hgail' in model_labels[i]:
            policy = utils.build_hierarchy(args, env)
        else:
            policy = utils.build_policy(args, env)

        # initialize variables
        sess.run(tf.global_variables_initializer())

        # load params
        if 'hgail' in model_labels[i]:
            for j, level in enumerate(policy):
                level.algo.policy.set_param_values(params[j]['policy'])
            policy = policy[0].algo.policy
        else:
            policy.set_param_values(params['policy'])

        # collect imgs
        if args.env_multiagent:
            imgs = mutliagent_simulate(
                env, 
                policy, 
                max_steps=max_steps, 
                env_kwargs=env_kwargs,
                render_kwargs=render_kwargs
            )
        else:
            imgs = simulate(
                env, 
                policy, 
                max_steps=max_steps, 
                env_kwargs=env_kwargs,
                render_kwargs=render_kwargs
            )
        render_map[model_labels[i]] = imgs
    return render_map


#-----------------------------------------------------------------------------
#			FUNCTION: Do it all once
#-----------------------------------------------------------------------------
def do_it_all_once(model_labels, model_args_filepaths, model_params_filepaths,
                   multi=False, name='single_multi', single_multi_comp=1, rand=None, n_vehs=None,
                  remove_ngsim=False):
    #do this with just 2 models at a time.
    print("creating render map for: ", "; ".join(model_labels))
    render_map = create_render_map(model_labels, model_args_filepaths, model_params_filepaths, multi,rand, n_vehs=n_vehs, remove_ngsim=remove_ngsim)
    

    # Raunak: Choose whether one or two at a time
    twoAtATime = False

    # Two models at a time
    if twoAtATime:
        imgs = [np.concatenate((a,b), 0) for (a,b) in zip(*[render_map[i] for i in model_labels])]
    else:
    # One model at a time
        imgs = [np.concatenate((a), 0) for (a) in zip(*[render_map[i] for i in model_labels])]
    
    
    fig, ax = plt.subplots(figsize=(16,16))
    plt.title(name)
    print("\nplotting")
    
    img = plt.imshow(imgs[0])

    def animate(i):
        img.set_data(imgs[i])
        return (img,)

    anim = animation.FuncAnimation(
        fig, 
        animate, 
        frames=len(imgs), 
        interval=100, 
        blit=True
    )

    WriterClass = animation.writers['ffmpeg']
    writer = WriterClass(fps=10, metadata=dict(artist='bww'), bitrate=1800)
    anim.save('../../data/media/' + name + '.mp4', writer=writer)
    print("Saved: ", name)

    HTML(anim.to_html5_video())
    plt.close()

#-----------------------------------------------------------------------------
#			The actual running thing
#-----------------------------------------------------------------------------
remove_ngsim_vehicles = False
for i in range(1):
    print("\Run number: ", i)
    seed = 8
    for j in [1]: #number of models to 'average'
        indx = (j-1)*2
        name = "-".join(model_labels[indx:indx+2])+'_'+str(i)+"_"+str(seed)
        if remove_ngsim_vehicles:
                name = name + '_ngsim_removed'
        do_it_all_once(model_labels[indx:indx+2], 
                       model_args_filepaths[indx:indx+2], 
                       model_params_filepaths[indx:indx+2], 
                       multi=True, 
                       name=name, 
                       single_multi_comp=j, 
                       rand=seed,
                       n_vehs=100,remove_ngsim=remove_ngsim_vehicles)
        print("\nDone once.\n")
