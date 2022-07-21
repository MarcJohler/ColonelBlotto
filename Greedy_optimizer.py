# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:04:55 2022

@author: Marc
"""
import numpy as np
import math
import scipy.special as ss
from Blotto_alpha_rank import blotto_mechanism, get_unique_permutations, plot_strategies, best_response_candidates, evaluate_strategy_subset, blotto_mechanism


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
                              preselect = False, initialization = "weights", max_support_size = 10, patience = 5, loss_goal = -1, plot_every = 10, surpress_plots = False):
    
    # compute budgets from strategies
    own_budget = sum(strategies[0])
    n_battlefields = strategies.shape[1]
    n_perms = ss.perm(n_battlefields, n_battlefields, True)
    n_strategies = strategies.shape[0]
    # check if battlefields are symmetric
    symmetric_battlefields = False
    if all(own_weights == opponent_weights) and len(np.unique(own_weights)) == 1:
        symmetric_battlefields = True
    
    if preselect:
        own_weights_order = np.argsort(own_weights)
        strategies_order = np.argsort(strategies, axis = 1)
        select = np.all(own_weights_order == strategies_order, axis = 1)
        own_strategies = strategies[select]
        n_strategies = own_strategies.shape[0]
        opponent_strategies = strategies
    
    else:
        own_strategies = strategies
        opponent_strategies = strategies
    
    
    # generate first bid from initialization
    if initialization == "weights":
        mixed_strategy = own_weights * own_budget / sum(own_weights)
        # exclude action in set if it is present
        remaining = np.sum(own_strategies == mixed_strategy, axis = 1) < n_battlefields
        #remaining_strategies = own_strategies[remaining]
        remaining_strategies = own_strategies
        n_remaining_strategies = n_strategies #- sum(remaining == False) 
        support_size = 1
        # add weights strategy to strategies if it is not present
        if sum(remaining == False) == 0:
            opponent_strategies = add_strategy(opponent_strategies, mixed_strategy, symmetric_battlefields, n_perms)
    elif initialization == "random":
        selected_index = np.random.choice(n_strategies)
        mixed_strategy = own_strategies[selected_index]
        #remaining_strategies = own_strategies[np.arange(0, n_strategies) != selected_index]
        remaining_strategies = own_strategies
        n_remaining_strategies = n_strategies #- 1
        support_size = 1
    elif initialization == "empty":
        mixed_strategy = None
        remaining_strategies = own_strategies
        n_remaining_strategies = n_strategies
        support_size = 0
    else:
        mixed_strategy = initialization
        remaining_strategies = own_strategies
        n_remaining_strategies = n_strategies
        support_size = mixed_strategy.shape[0]
    
    if symmetric_battlefields and initialization != "empty":
        mixed_strategy = get_unique_permutations(mixed_strategy, n_perms)
        
    
    # save best strategy set and its loss
    overall_best_set = mixed_strategy
    overall_best_loss = evaluate_strategy_subset(opponent_strategies, mixed_strategy, own_weights, opponent_weights, opponent_budget, tie_breaking_rule, learnable_return = False)
    remaining_patience = patience
    while support_size < max_support_size:
        iteration_best_idx = None
        iteration_best_set = None
        iteration_best_loss = 1
        maximum_loss = True
        constant_loss = True
        for idx, strategy in enumerate(remaining_strategies):
            new_strategy = add_strategy(mixed_strategy, strategy, symmetric_battlefields, n_perms)
            loss = evaluate_strategy_subset(opponent_strategies, new_strategy, own_weights, opponent_weights, opponent_budget, tie_breaking_rule, learnable_return = False)
            # check if loss improved
            if loss < iteration_best_loss:
                if not maximum_loss:
                    constant_loss = False
                iteration_best_loss = loss
                iteration_best_idx = idx
                iteration_best_set = new_strategy
                maximum_loss = False
            elif loss > iteration_best_loss:
                constant_loss = False
        # if loss is constant over whole iteration choose a random strategy
        if constant_loss:
            iteration_best_idx = np.random.choice(n_remaining_strategies)
            iteration_best_set = add_strategy(mixed_strategy, remaining_strategies[iteration_best_idx], symmetric_battlefields, n_perms)
            mixed_strategy = iteration_best_set
        else:
            # save best strategy set of iteration
            mixed_strategy = iteration_best_set
        # check if there is an overall improvement compared to the current best strategy
        if iteration_best_loss >= overall_best_loss:
            remaining_patience -= 1
        else:
            overall_best_loss = iteration_best_loss
            overall_best_set = iteration_best_set
            remaining_patience = patience
        # exclude chosen strategy from remaining strategies
        #remaining_strategies = remaining_strategies[np.arange(0, n_remaining_strategies) != iteration_best_idx]
        #remaining_strategies = own_strategies
        #n_remaining_strategies -= 1
        support_size += 1
        print(mixed_strategy, "\n")
        
        print("Best loss of sets with support size ", support_size, ": ", iteration_best_loss)
        print("Best overall loss: ", overall_best_loss)
        print("Remaining patience: ", remaining_patience, "\n")
        # if there is no more patience remaining, stop
        if remaining_patience == 0:
            break
    
    return overall_best_set, overall_best_loss


def greedy_strategy_optimizer_backward(strategies, opponent_budget = None, own_weights = None, opponent_weights = None, tie_breaking_rule = "right-in-two",
                              restarts = 1, max_support_size = 20, sample_support_size = 40, patience = 5, loss_goal = -1, plot_every = 10, surpress_plots = False):
    # compute budgets from strategies
    own_budget = sum(strategies[0])
    n_battlefields = strategies.shape[1]
    n_perms = ss.perm(n_battlefields, n_battlefields, True)
    n_strategies = strategies.shape[0]
    # check if battlefields are symmetric
    symmetric_battlefields = False
    if all(own_weights == opponent_weights) and len(np.unique(own_weights)) == 1:
        symmetric_battlefields = True
    
    opponent_strategies = strategies
    # save best strategy set and its loss
    overall_best_set = strategies
    overall_best_indizes = np.array([])
    overall_best_loss = 1
    remaining_patience = patience
    for i in range(restarts):
        if sample_support_size == "max":
            own_strategies = strategies
            support_size = n_strategies
        else:
            own_strategies = strategies[np.random.choice(n_strategies, sample_support_size, replace = False)]
            support_size = sample_support_size
            
        # compute loss and best response for full set
        loss, best_response = evaluate_strategy_subset(opponent_strategies, own_strategies, own_weights, opponent_weights, opponent_budget, tie_breaking_rule, return_best_response = True)
        # let the whole set play against its best response again to find candidates to exclude
        results_against_best_response = np.zeros(support_size)
        for idx, strategy in enumerate(own_strategies):
            results_against_best_response[idx] = blotto_mechanism(best_response, strategy, opponent_weights, own_weights, tie_breaking_rule)
            
        # start greedy optimizing
        while support_size > 1:
            iteration_best_idx = None
            iteration_best_set = None
            iteration_best_loss = 1
            best_response_to_best_set = None 
            max_loss_against_best_response = max(results_against_best_response)
            maximum_loss = True
            constant_loss = True
            
            for idx, strategy in enumerate(own_strategies):
                if results_against_best_response[idx] < max_loss_against_best_response:
                    continue
                new_strategy = own_strategies[np.arange(support_size) != idx]
                loss, best_response = evaluate_strategy_subset(opponent_strategies, new_strategy, own_weights, opponent_weights, opponent_budget, tie_breaking_rule, return_best_response = True)
                # check if loss improved
                if loss < iteration_best_loss:
                    if not maximum_loss:
                        constant_loss = False
                    iteration_best_loss = loss
                    iteration_best_idx = idx
                    iteration_best_set = new_strategy
                    best_response_to_best_set = best_response
                    maximum_loss = False
                elif loss > iteration_best_loss:
                    constant_loss = False
            # if loss is constant over whole iteration choose a random strategy
            if constant_loss:
                iteration_best_idx = np.random.choice(np.arange(support_size)[results_against_best_response == max_loss_against_best_response])
                iteration_best_set = own_strategies[np.arange(support_size) != iteration_best_idx]
                loss, best_response_to_best_set = evaluate_strategy_subset(opponent_strategies, iteration_best_set, own_weights, opponent_weights, opponent_budget, tie_breaking_rule, return_best_response = True)
                own_strategies = iteration_best_set
            else:
                # save best strategy set of iteration
                own_strategies = iteration_best_set
            
            # Let the best set of the iteration play again against its best response in order to find actions which could be excluded in the next iteration
            results_against_best_response = np.zeros(support_size - 1)
            for idx, strategy in enumerate(iteration_best_set):
                results_against_best_response[idx] = blotto_mechanism(best_response_to_best_set, strategy, opponent_weights, own_weights, tie_breaking_rule)
            
            # check if there is an overall improvement compared to the current best strategy
            if iteration_best_loss >= overall_best_loss:
                remaining_patience -= 1
            else:
                overall_best_indizes = np.append(overall_best_indizes, iteration_best_idx)
                overall_best_loss = iteration_best_loss
                overall_best_set = iteration_best_set
                remaining_patience = patience
            support_size -= 1
        
            #print(own_strategies)
            print("Best loss of sets with support size ", support_size, ": ", iteration_best_loss)
            print("Best overall loss: ", overall_best_loss)
            print("Remaining patience: ", remaining_patience, "\n")
            
            # if there is no more patience remaining, stop
            if remaining_patience <= 0 or overall_best_loss <= loss_goal: 
                break
        if remaining_patience <= 0 or overall_best_loss <= loss_goal:
            break
    
    return overall_best_set, overall_best_indizes, overall_best_loss
    
        
    