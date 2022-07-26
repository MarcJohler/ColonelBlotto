# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 18:37:50 2022

@author: marc
"""
import numpy as np
from Blotto_ultimative_validation import blotto_ultimative_validation
from Blotto_discretizer import discretize_action_space

number_of_battlefields = 3
budget1 = 1000
budget2 = 1500
granularity_level = 32
add_noise = False
batch_size = 10
outer_epochs = 10**3
inner_epochs = 10**6
mode = "kmeans"
eval_every = 10**3
patience = 10**3
loss_goal = 0.4
mr = 0.1
pop_size = 10
restarts = 1
tie_breaking_rule = "right-in-two"
weights1 = np.array([1, 1, 1])
symmetric_battlefields = len(np.unique(weights1)) == 1

strategies, probs = discretize_action_space(number_of_battlefields, budget1, symmetric_battlefields, 
                                            granularity_level = granularity_level, add_noise = add_noise, integer_bids = True)

##############
strategies11_1, probs11_1 = discretize_action_space(number_of_battlefields, budget1, symmetric_battlefields, 
                                                    granularity_level = 11, add_noise = add_noise, integer_bids = True)
##############
strategies49_1, probs49_1 = discretize_action_space(number_of_battlefields, budget1, symmetric_battlefields, 
                                                    granularity_level = 49, add_noise = add_noise, integer_bids = True)
##############
strategie25_1, probs25_1 = discretize_action_space(number_of_battlefields, budget1, symmetric_battlefields, 
                                                    granularity_level = 25, add_noise = add_noise, integer_bids = True)
##############
strategies32_1, probs32_1 = discretize_action_space(number_of_battlefields, budget1, symmetric_battlefields, 
                                                    granularity_level = 32, add_noise = add_noise, integer_bids = True)
##############
strategies27_1, probs27_1 = discretize_action_space(number_of_battlefields, budget1, symmetric_battlefields, 
                                                    granularity_level = 27, add_noise = add_noise, integer_bids = True)

combined = np.vstack((strategies11_1, strategies49_1, strategie25_1, strategies32_1, strategies27_1)).astype(int)
combined = np.unique(combined, axis = 0)

probs = None

all_time_loss, action_frequencies = blotto_ultimative_validation(strategies, probs, strategies2 = None, probs2 = None, weights1 = weights1, symmetric_battlefields = symmetric_battlefields,
                      pop_size = pop_size, alpha = 100, mr = mr, tie_breaking_rule = tie_breaking_rule,
                      batch_size = batch_size, restarts = restarts, outer_epochs = outer_epochs, inner_epochs = inner_epochs, 
                      ordered_output = True, track_every = inner_epochs * 10, eval_mode = "kmeans", eval_every = eval_every, 
                      patience = 10**5, loss_goal = loss_goal, plot_every = inner_epochs * 10, surpress_plots = True)

# compute metrics
nash_eq = np.sum(strategies > 2 / number_of_battlefields * budget1, axis = 1) == 0
# compute True positives, True negatives, False positives, False negatives
TP = np.sum((action_frequencies > 0) * nash_eq)
TN = np.sum((action_frequencies == 0) * (nash_eq == False))
FP = np.sum((action_frequencies > 0) * (nash_eq == False))
FN = np.sum((action_frequencies == 0) * nash_eq)
# compute sensitivity, specifity and balanced accuracy
sensitivity = TP / (TP + FN)
specifity = TN / (TN + FP)
balanced_accuracy = 0.5 * sensitivity + 0.5 * specifity  
# compute percentage of actions played from NE
nash_percentage = np.sum(nash_eq * action_frequencies) / np.sum(action_frequencies)