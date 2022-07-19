# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:04:55 2022

@author: Marc
"""
import numpy as np
import math
import scipy.special as ss
from Blotto_alpha_rank import blotto_mechanism, get_unique_permutations, plot_strategies, best_response_candidates, evaluate_strategy_subset


def add_strategy(strategy_set, add, symmetric_battlefields, n_perms):
    if strategy_set is None and symmetric_battlefields:
        return get_unique_permutations(add, n_perms)
    elif strategy_set is None:
        return add.reshape(1, len(add))
    elif symmetric_battlefields:
        return np.vstack((strategy_set, get_unique_permutations(add, n_perms)))
    else:
        return np.vstack((strategy_set, add))
        

def greedy_strategy_optimizer(strategies, opponent_budget = None, own_weights = None, opponent_weights = None, tie_breaking_rule = "right-in-two",
                              initialization = "weights", max_support_size = 10, patience = 5, loss_goal = -1, plot_every = 10, surpress_plots = False):
    
    # compute budgets from strategies
    own_budget = sum(strategies[0])
    n_battlefields = strategies.shape[1]
    n_perms = ss.perm(n_battlefields, n_battlefields, True)
    n_strategies = strategies.shape[0]
    # check if battlefields are symmetric
    symmetric_battlefields = False
    if all(own_weights == opponent_weights) and len(np.unique(own_weights)) == 1:
        symmetric_battlefields = True
    
    # generate first bid from initialization
    if initialization == "weights":
        mixed_strategy = own_weights * own_budget / sum(own_weights)
        # exclude action in set if it is present
        remaining = np.sum(strategies == mixed_strategy, axis = 1) < n_battlefields
        remaining_strategies = strategies[remaining]
        n_remaining_strategies = n_strategies - sum(remaining == False) 
        support_size = 1
        # add weights strategy to strategies if it is not present
        if sum(remaining == False) == 0:
            strategies = add_strategy(strategies, mixed_strategy, symmetric_battlefields, n_perms)
            
        
    elif initialization == "random":
        selected_index = np.random.choice(n_strategies)
        mixed_strategy = strategies[selected_index]
        remaining_strategies = strategies[np.arange(0, n_strategies) != selected_index]
        n_remaining_strategies = n_strategies - 1
        support_size = 1
    elif initialization == "empty":
        mixed_strategy = None
        remaining_strategies = strategies
        n_remaining_strategies = n_strategies
        support_size = 0
    
    if symmetric_battlefields and initialization != "empty":
        mixed_strategy = get_unique_permutations(mixed_strategy, n_perms)
        
    # save best strategy set and its loss
    overall_best_set = None
    overall_best_loss = math.inf
    remaining_patience = patience
    while 0 < max_support_size:
        iteration_best_idx = None
        iteration_best_set = None
        iteration_best_loss = math.inf
        constant_loss = True
        for idx, strategy in enumerate(remaining_strategies):
            new_strategy = add_strategy(mixed_strategy, strategy, symmetric_battlefields, n_perms)
            loss = evaluate_strategy_subset(strategies, new_strategy, own_weights, opponent_weights, opponent_budget, tie_breaking_rule, learnable_return = True)
            # check if loss improved
            if loss < iteration_best_loss:
                iteration_best_loss = loss
                iteration_best_idx = idx
                iteration_best_set = new_strategy
                constant_loss = False
            elif loss > iteration_best_loss:
                constant_loss = False
        # save best strategy set of iteration
        mixed_strategy = iteration_best_set
        # if loss is constant over whole iteration choose a random strategy
        if constant_loss:
            iteration_best_idx = np.random.choice(n_remaining_strategies)
            iteration_best_set = add_strategy(mixed_strategy, remaining_strategies[iteration_best_idx], symmetric_battlefields, n_perms)
        # check if there is an overall improvement compared to the current best strategy
        if iteration_best_loss >= overall_best_loss:
            remaining_patience -= 1
        else:
            overall_best_loss = iteration_best_loss
            overall_best_set = iteration_best_set
            remaining_patience = patience
        # exclude chosen strategy from remaining strategies
        #remaining_strategies = remaining_strategies[np.arange(0, n_remaining_strategies) != iteration_best_idx]
        remaining_strategies = strategies
        #n_remaining_strategies -= 1
        max_support_size = 30
        support_size += 1
        print("Best loss of sets with support size ", support_size, ": ", iteration_best_loss)
        print("Best overall loss: ", overall_best_loss)
        print("Remaining patience: ", remaining_patience, "\n")
        # if there is no more patience remaining, stop
        if remaining_patience == 0:
            break
    
    return overall_best_set, overall_best_loss

        
    