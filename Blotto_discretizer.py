#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:50:48 2022

@author: marc
"""

import itertools
import math
import numpy as np
import scipy.special as ss

def discretize_action_space(number_of_battlefields, budget, symmetric_battlefields = True, granularity_level = None, add_noise = False, integer_bids = True):
    if granularity_level is None:
        granularity_level = number_of_battlefields
    
    combinations = itertools.combinations_with_replacement(range(number_of_battlefields), granularity_level)
    n_combinations = ss.comb(number_of_battlefields, granularity_level, exact = True, repetition = True)
    
    pure_strategies = np.zeros((n_combinations, number_of_battlefields))
     
    granularity = budget / granularity_level 
    
    max_noise = 0
    if add_noise:
        max_noise = math.floor(granularity / number_of_battlefields)
    
    for i, comb in enumerate(combinations):
        for idx in comb:
            # add a proportion of the budget as bid to the strategy i
            pure_strategies[i, idx] += granularity
        
        if symmetric_battlefields:
            pure_strategies[i, ] = np.sort(pure_strategies[i, ])
        
        # don't use with symmetric_battlefields = True, will fix this soon
        if add_noise:
            # add random noise without enabling changing the ordinal structure of bids
            sign = 2 * np.random.binomial(1, 0.5, number_of_battlefields) - 1
            noise = np.random.binomial(max_noise, 0.5, number_of_battlefields)
            perturbed_strategy = pure_strategies[i, ] + sign * noise
            # invert negative bids
            perturbed_strategy  -= 2 * perturbed_strategy * (perturbed_strategy  < 0)
            # rescale sum to budget
            pure_strategies[i, ]  = perturbed_strategy  / sum(perturbed_strategy) * budget
        if integer_bids:
            # compute the partial units which shall be distributed
            partial_units = int(round(sum(pure_strategies[i, ] % 1)))
            if partial_units > 0:
                # add to the 'partial_units' highest bids to preserve the order
                add_here = np.argsort(pure_strategies[i, ])[-partial_units:]
                pure_strategies[i, add_here] += 1
                # truncate the decimals to get integer bids
                pure_strategies = np.trunc(pure_strategies)
    
    probs = "uniform"
    # if battlefields are symmetric only return unique splits
    if symmetric_battlefields:
        pure_strategies, probs = np.unique(pure_strategies, axis = 0, return_counts = True)
        probs = probs / np.sum(probs)
        
    return pure_strategies, probs
    

#n_strategies = pure_strategies1.shape[0]
##############
#win_counter = np.zeros((n_strategies, 2))
#for i in range(n_strategies):
    #strategy_i = pure_strategies1[i, ]
    #for j in range(n_strategies):
        #if i == j:
            #next
        #strategy_j = pure_strategies1[j, ]
        #fields_won = sum(strategy_i >= strategy_j)
        #fields_lost = sum(strategy_i <= strategy_j)
        #if fields_won > fields_lost:
            #win_counter[i, 0] += 1
        #elif fields_won == fields_lost:
            #win_counter[i, 1] += 1
#############