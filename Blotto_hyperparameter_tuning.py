#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 19:37:45 2022

@author: marc
"""

from Blotto_discretizer import discretize_action_space
from Blotto_alpha_rank import blotto_alpha_rank
import numpy as np
import multiprocessing as mp
from multiprocessing import set_start_method
import sys
from numba import cuda
import torch

number_of_battlefields = 3
budget1 = 1000
budget2 = 1000
granularity_level = 10
add_noise = False
epochs = 10**7
mode = "kmeans"
eval_every = 10**2
patience = 10**4
restarts = 1
track_every = 10**5
tie_breaking_rule = "right-in-two"
weights1 = np.array([1, 1, 1])
symmetric_battlefields = len(np.unique(weights1)) == 1

#cuda.select_device(gpu)

strategies1, probs1 = discretize_action_space(number_of_battlefields, budget1, symmetric_battlefields, 
                                             granularity_level, add_noise = add_noise, integer_bids = True)

probs1 = "uniform"
strategies2, probs2 = None, None
#strategies2, probs2 = discretize_action_space(number_of_battlefields, budget2, symmetric_battlefields, 
#                                             granularity_level, add_noise = add_noise, integer_bids = True)

#mrs = [5**i for i in range(-6, -1)]
#pop_sizes = [10*i for i in range(1, 6)]

input_tuples = [(5*i, 10**(j / 2), False) for i in range(1, 7) for j in range(-7, -1)]

# set one tuple for progress display (find a better solution for this later)
progress_tuple = list(input_tuples[-1])
progress_tuple[-1] = True
input_tuples[-1] = tuple(progress_tuple)

# helps to apply function in multiprocessing
def helper_func(pop_size, mr, track_progress):
    return blotto_alpha_rank(strategies1, probs1, strategies2, probs2, weights1 = weights1, weights2 = None, tie_breaking_rule = tie_breaking_rule, 
                             pop_size = pop_size, alpha = 100, mr = mr, restarts = restarts, epochs = epochs, 
                             track_every = track_every, eval_mode = mode, eval_every = eval_every, patience = patience, plot_every = epochs * 10)

# converts multiprocessing output to dictionary
def output_to_dict(val_list, input_tuples):
    output_dict = {}
    for i, input in enumerate(input_tuples):
        output_dict[input] = val_list[i]
    return output_dict

pool_obj = mp.Pool(processes = 10)

#set_start_method("spawn")
output = pool_obj.starmap(helper_func, input_tuples)
output_dict = output_to_dict(output, input_tuples)
            
# write into textfile
w = open(sys.path[0] + "/outputs/output_non_noise_granularity" + str(granularity_level) + "test.txt", "w")
w.write(str(output_dict))
w.close()

#torch.cuda.empty_cache()