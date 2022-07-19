# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:18:56 2022

@author: Marc
"""

from Blotto_discretizer import discretize_action_space
from Greedy_optimizer import greedy_strategy_optimizer
import numpy as np

number_of_battlefields = 4
budget1 = 1000
budget2 = 1000
granularity = 10
add_noise = False
patience = 10
tie_breaking_rule = "right-in-two"
max_support_size = 30
weights1 = np.array([1, 1, 2, 2])
symmetric_battlefields = len(np.unique(weights1)) == 1

one_granularity, probs = discretize_action_space(number_of_battlefields, budget1, symmetric_battlefields = symmetric_battlefields, 
                                              granularity_level = granularity, add_noise = add_noise, integer_bids = False)

#cuda.select_device(gpu)
strategies11, probs11 = discretize_action_space(number_of_battlefields, budget1, symmetric_battlefields = symmetric_battlefields, 
                                              granularity_level = 11, add_noise = add_noise, integer_bids = False)

strategies9, probs9 = discretize_action_space(number_of_battlefields, budget1, symmetric_battlefields = symmetric_battlefields, 
                                              granularity_level = 9, add_noise = add_noise, integer_bids = False)

strategies13, probs13 = discretize_action_space(number_of_battlefields, budget1, symmetric_battlefields, 
                                              granularity_level = 13, add_noise = add_noise, integer_bids = False)

strategies5, probs5 = discretize_action_space(number_of_battlefields, budget1, symmetric_battlefields, 
                                              granularity_level = 5, add_noise = add_noise, integer_bids = False)

strategies7, probs7 = discretize_action_space(number_of_battlefields, budget1, symmetric_battlefields, 
                                              granularity_level = 7, add_noise = add_noise, integer_bids = False)

combined = np.vstack((strategies11, strategies13, strategies5, strategies7))
# get rid of unwanted duplicates
combined = np.unique(np.round(combined, 4), axis = 0)


best_set, best_loss = greedy_strategy_optimizer(one_granularity, opponent_budget = budget2, own_weights = weights1, opponent_weights = weights1, tie_breaking_rule = "right-in-two",
                                                initialization = "empty", max_support_size = max_support_size, patience = patience, loss_goal = -1, plot_every = 10, surpress_plots = False)