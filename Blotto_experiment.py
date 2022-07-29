#%%

from Blotto_discretizer import discretize_action_space
from Blotto_alpha_rank import blotto_alpha_rank
import numpy as np
import multiprocessing as mp
from numba import jit, njit, cuda
import sys
#import torch

number_of_battlefields = 3
budget1 = 1000
budget2 = 1000
granularity_level = 18
add_noise = False
epochs = 10**7
mode = "extensive"
eval_every = 10**2
patience = 10**8
mr = 0.03162277660168379
pop_size = 5
restarts = 1
plot_every = 10**6
tie_breaking_rule = "right-in-two"
weights1 = np.array([1, 1, 1])
symmetric_battlefields = len(np.unique(weights1)) == 1

#cuda.select_device(gpu)

strategies1, probs1 = discretize_action_space(number_of_battlefields, budget1, symmetric_battlefields, 
                                              granularity_level = granularity_level, add_noise = add_noise, integer_bids = True)
# activate uniform distribution since the other approach doesn't work
probs1 = "uniform"

#strategies2, probs2 = None, None
strategies2, probs2 = discretize_action_space(number_of_battlefields, budget2, symmetric_battlefields,
                                              granularity_level = granularity_level, add_noise = add_noise, integer_bids = True)

probs2 = "uniform"
#@njit(nopython = True)
#def helper_function():
#   return blotto_alpha_rank(strategies1, probs1, strategies2, probs2, pop_size = pop_size, alpha = 0, mr = mr, 
#                          restarts = restarts, epochs = epochs, 
#                          track_progress = True, evaluate_every = evaluate_every, plot_every = True)



ranks, labels = blotto_alpha_rank(strategies1, probs1, strategies2, probs2, weights1 = weights1, weights2 = None, tie_breaking_rule = tie_breaking_rule, 
                                  pop_size = pop_size, alpha = 100, mr = mr, restarts = restarts, epochs = epochs, 
                                  track_every = plot_every, eval_mode = mode, eval_every = eval_every, patience = patience, plot_every = plot_every)



w = open(sys.path[0] + "/outputs/blotto_experiment_ranks_extensive_gran18.txt", "w")
w.write(str(ranks))
w.close()

w2 = open(sys.path[0] + "/outputs/blotto_experiment_labels_extensive_gran18.txt", "w")
w2.write(str(labels))
w2.close()
# %%
