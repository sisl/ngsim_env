import numpy as np
import os

import utils
import copy # for combine_trajs

const_default_models = [
    'singleagent_1',
    'multiagent_1',
    'singleagent_2',
    'multiagent_2',
    'singleagent_3',
    'multiagent_3'
]

def create_val_dir_str(base='singleagent', basedir='../../data/experiments/'):
    return os.path.join(basedir, base, 'imitate/validation/')

def get_trajs_dict(model_labels = const_default_models, basedir='../../data/experiments/', files_to_use=[0,1,2,3,4,5]):
    traj_lab_dict = dict()
    for label in model_labels:
        traj_lab_dict[label] = utils.load_trajs_labels(create_val_dir_str(label, basedir), files_to_use=files_to_use)
    return traj_lab_dict


def get_val_dirs_and_params_paths(model_labels = const_default_models,
                                 itr_n = 1000,
                                 basedir = '../../data/experiments'):
    valdirs = [create_val_dir_str(label, basedir) for label in model_labels]
    params_filepaths = [os.path.join(basedir, label, 'imitate/log/itr_' + str(itr_n) + '.npz') for label in model_labels]
    return valdirs, params_filepaths

#pass a dict of iters to allow different iterations for each model. - needed before fine tuning.
def get_val_dirs_and_params_paths_d(model_labels = const_default_models,
                                 itr_n = {},
                                 basedir = '../../data/experiments'):
    valdirs = [create_val_dir_str(label, basedir) for label in model_labels]
    params_filepaths = [os.path.join(basedir, label, 'imitate/log/itr_' + str(itr_n[label]) + '.npz') for label in model_labels]
    return valdirs, params_filepaths


'''
This function creates creates as single trajectory that is the concatenation of all the others.
 The purpose is to serve as an average trajectory of sorts, over all model runs (of a given type - multi vs single)
 Imagine a list of N trajectories each 100 x dict of (200 x 50) items.
 The idea here is to return a single trajectory of (N*100) x dict of (200 x 50) items.
 In our context, it would be N models each with S scenes, each a dict of (H horizon x V vehicles) input,
 and we output N*S scenes, each a dict of (H horizon x V vehicles).
'''
def combine_trajs(list_of_trajs):
    result = copy.deepcopy(list_of_trajs[0])
    n_datasets = len(list_of_trajs[0])
    for ds in range(n_datasets):
        for scene_j in range(len(result[ds])):
            for other_traj in range(1,len(list_of_trajs)): # skip the first, since we already did that
                result[ds] = np.concatenate((result[ds], list_of_trajs[other_traj][ds]))
            #for attr in result[ds][scene_j].keys():
            #    for other_traj in range(1,len(list_of_trajs)): # skip the first, since we already did that
            #        result[ds][scene_j][attr] = np.concatenate((result[ds][scene_j][attr],
            #                                                    list_of_trajs[other_traj][ds][scene_j][attr]), axis=1)
    return result