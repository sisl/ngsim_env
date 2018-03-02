import matplotlib
matplotlib.use('agg')
import pylab as plt
import numpy as np
import os
import sys

import utils
import visualize_utils

file_name_id = ""
if len(sys.argv) > 1:
    file_name_id = str(sys.argv[1])

model_labels = [
    'singleagent_def_1_fine',
    'singleagent_def_2_fine',
    'singleagent_def_3_fine',
    'multiagent_curr_1_fine',
    'multiagent_curr_2_fine',
    'multiagent_curr_3_fine'
]

print("Loading model trajectories...", flush=True)
traj_lab_dict = visualize_utils.get_trajs_dict(model_labels)
print("Done.", flush=True)

def visualize_trajectories(trajs, length, label='', color='blue', attr='rmse', style='solid'):
    rmses = []
    for traj in trajs:
        if len(traj[attr]) == length:
            #print(traj[attr].shape) # 200 x 50
            #print(np.mean(traj[attr],axis=1).shape) # 200 x 1
            # here we want to simply act as though all of the trajectories are independent. 
            # We no longer care that they were from the same scene.
            for veh_i in range(traj[attr].shape[1]):
                rmses.append(traj[attr][:,veh_i])
    rmses = np.array(rmses)
    mean = np.mean(rmses, axis=0)
    bound = np.std(rmses, axis=0) / np.sqrt(len(rmses)) / 2
    x = range(len(mean))
    plt.fill_between(x, mean - bound, mean + bound, alpha=.4, color=color)
    plt.plot(x, mean, c=color, label='mean {}: {:.5f}'.format(attr, np.mean(rmses)), linestyle=style)
    plt.xlabel('timesteps')
    plt.ylabel(attr)
    plt.title(label)
    plt.legend()

def plot_validation(trajs, labels, color='blue', length=200, attr='rmse', style='solid'):
    trajs = [
        trajs[0],
        np.concatenate((trajs[1], trajs[2])),
        np.concatenate((trajs[3], trajs[4], trajs[5]))
    ]
    labels = [
        labels[0],
        labels[1] + ' ' + labels[2],
        labels[3] + ' ' + labels[4] + ' ' + labels[4] 
    ]
    for i, traj in enumerate(trajs):
        plt.subplot(1,3,i+1)
        visualize_trajectories(traj, length, labels[i], attr=attr, color=color, style=style)


attrs = ['rmse_pos', 'rmse_t', 'rmse_vel', 'is_colliding']
colors = {
    'singleagent_def_1_fine_': 'blue',
    'singleagent_def_2_fine_': 'red',
    'singleagent_def_3_fine_': 'magenta',
    'multiagent_curr_1_fine_': 'blue',
    'multiagent_curr_2_fine_': 'red',
    'multiagent_curr_3_fine_': 'magenta',
    'singleagent_def_1_fine': 'blue',
    'singleagent_def_2_fine': 'red',
    'singleagent_def_3_fine': 'magenta',
    'multiagent_curr_1_fine': 'blue',
    'multiagent_curr_2_fine': 'red',
    'multiagent_curr_3_fine': 'magenta'
}
for i, label in enumerate(model_labels):
    style = 'dotted'
    if 'multi' in label: style = 'solid'
    print(label, colors[label], style, flush=True)


length = 200
ngsim_labels = traj_lab_dict[model_labels[0]][1]
for attr in attrs:
    plt.figure(figsize=(16,4))
    for i, label in enumerate(model_labels):
        style = 'dotted'
        if 'multi' in label: style = 'solid'
        plot_validation(traj_lab_dict[label][0], ngsim_labels, 
                    color=colors[label], attr=attr, style=style, length=length)
    filename = attr + '_' + file_name_id + '.png'
    plt.savefig(filename)
    print("Saved:", filename, flush=True)

    

# averages
all_multi_trajs = visualize_utils.combine_trajs([traj_lab_dict[model_label][0] \
				for model_label in model_labels if 'multi' in model_label])
all_single_trajs = visualize_utils.combine_trajs([traj_lab_dict[model_label][0] \
				for model_label in model_labels if 'single' in model_label])


for attr in attrs:
    plt.figure(figsize=(16,4))
    plot_validation(all_single_trajs, ngsim_labels, color='black', attr=attr, style='dotted', length=length)
    plot_validation(all_multi_trajs, ngsim_labels, color='black', attr=attr, style='solid', length=length)

    filename = attr + '_avg_' + file_name_id + '.png'
    plt.savefig(filename)
    print("Saved:", filename, flush=True)

