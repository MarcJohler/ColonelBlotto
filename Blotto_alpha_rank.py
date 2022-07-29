#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 18:24:55 2022

@author: marc
"""

import numpy as np
import math
import warnings
import scipy.special as ss
import itertools
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
from numba import cuda, njit, jit

# computes the utility for the first player in a blotto game
#@njit(nopython = True)
def blotto_mechanism(bid1, bid2, weights1 = None, weights2 = None, tie_breaking_rule = "right-in-two", learnable_return = False):
    assert tie_breaking_rule in ["right-in-two", "all-or-nothing", "no-reward", "sharing-is-caring"]
    # if weights1 is not defined assume uniform weights
    if weights1 is None:
        weights1 = np.ones(len(bid1))
    # if weights2 is not defined assume equivalence to weights1
    if weights2 is None:
        weights2 = weights1
    # compute scores according to tie-breaking-rule
    if tie_breaking_rule == "no-reward":
        p1wins = sum(weights1[bid1 > bid2])
        p2wins = sum(weights2[bid1 < bid2])
    elif tie_breaking_rule == "sharing-is-caring":
        p1wins = sum(weights1[bid1 >= bid2])
        p2wins = sum(weights2[bid1 <= bid2])
    elif tie_breaking_rule == "right-in-two":
        equal = bid1 == bid2 
        p1wins = sum(weights1[bid1 > bid2]) + sum(1/2 * weights1[equal])
        p2wins = sum(weights2[bid2 > bid1]) + sum(1/2 * weights2[equal])
    elif tie_breaking_rule == "all-or-nothing":
        p1_fixed_wins = bid1 > bid2
        equal = bid1 == bid2
        p1_random_wins = equal * np.random.choice(2, len(equal))
        p1wins = (p1_fixed_wins + p1_random_wins) == 1
        p2wins = p1wins == False
        # include weights
        p1wins = sum(weights1[p1wins])
        p2wins = sum(weights2[p2wins])
    # compare scores and comopute payoffs for p1 
    if learnable_return:
        return((p1wins - p2wins) / (p1wins + p2wins))
    if p1wins == p2wins:
        return(0)
    elif p1wins > p2wins:
        return(1)
    else:
        return(-1)
    
# function to get unique permutations of array-like
#@njit(nopython = True)
def get_unique_permutations(arr, n_perms):
    perm_object = itertools.permutations(arr)
    perm_array = np.zeros((n_perms, len(arr)))
    # save as array
    for i, perm in enumerate(perm_object):
        perm_array[i] = np.array(perm)
    # return only unique permutation
    return np.unique(perm_array, axis = 0)

# function for selecting the current primary action in alpha rank
#@njit(nopython = True)
def selection_process(strategies, probs, primary, secondary, opponent_strat, 
                      weights1, weights2, tie_breaking_rule, alpha, mr, max_batch_size = 100):
    # check if battlefields are symmetric
    symmetric_battlefields = False
    if all(weights1 == weights2) and all(weights1 == 1):
        symmetric_battlefields = True
    # check if uniform distribution is used
    # fix this
    if np.all(probs == "uniform"):
        probs = None
    # mutate with a very small probability
    if np.random.binomial(1, mr):
        tmp = primary
        while tmp == primary:
            tmp = np.random.choice(strategies.shape[0], p = probs)
        primary = tmp
    elif primary == secondary:
        # if both are the same no need for computations
        pass
    else:
        # compare primary and secondary strategy in terms of fitness
        prim_strat = strategies[primary, ]
        sec_strat = strategies[secondary, ]
        
        if symmetric_battlefields:
            n_battlefields = len(prim_strat)
            # compute number of permutations
            strategy_perms = ss.perm(n_battlefields, n_battlefields, True)
            # if number of permutations exceeds max_batch_size sample matchups
            if strategy_perms**2 > max_batch_size:
                # initialize strategies
                prim_strat_shuffled = prim_strat
                sec_strat_shuffled = sec_strat
                opponent_strat_shuffled = opponent_strat
                # save results in arrays
                prim_fitness = 0
                sec_fitness = 0
                for i in range(max_batch_size):
                    # draw random permutation of each strategy
                    prim_strat_shuffled = np.random.shuffle(prim_strat_shuffled)
                    sec_strat_shuffled = np.random.shuffle(sec_strat_shuffled)
                    opponent_strat_shuffled = np.random.shuffle(opponent_strat_shuffled)
                    
                    prim_fitness += blotto_mechanism(prim_strat_shuffled, opponent_strat_shuffled, weights1, weights2, tie_breaking_rule)
                    sec_fitness += blotto_mechanism(sec_strat_shuffled, opponent_strat_shuffled, weights1, weights2, tie_breaking_rule)
                # compute fitness for prim and sec 
                prim_fitness = prim_fitness / max_batch_size
                sec_fitness = sec_fitness / max_batch_size
            else:    
                # otherwise iterate over all matchups and compute fitness 
                prim_perms = get_unique_permutations(prim_strat, strategy_perms)
                sec_perms = get_unique_permutations(sec_strat, strategy_perms)
                opponent_perms = get_unique_permutations(opponent_strat, strategy_perms)
                # save number of unique permutations
                n_prim = prim_perms.shape[0]
                n_sec = sec_perms.shape[0]
                n_opponent = opponent_perms.shape[0]
                # save results in arrays
                prim_fitness = 0
                sec_fitness = 0
                
                for opponent_id in range(n_opponent):
                    opponent_perm = opponent_perms[opponent_id]
                    for prim_id in range(n_prim):
                        prim_perm = prim_perms[prim_id]
                        prim_fitness += blotto_mechanism(prim_perm, opponent_perm, weights1, weights2, tie_breaking_rule)
                    
                    for sec_id in range(n_sec):
                        sec_perm = sec_perms[sec_id]
                        sec_fitness += blotto_mechanism(sec_perm, opponent_perm, weights1, weights2, tie_breaking_rule)
                # compute fitness for prim and sec
                prim_fitness = prim_fitness / (n_prim * n_opponent)
                sec_fitness = sec_fitness / (n_sec * n_opponent)
        else:
            prim_fitness = blotto_mechanism(prim_strat, opponent_strat, weights1, weights2, tie_breaking_rule)
            sec_fitness = blotto_mechanism(prim_strat, opponent_strat, weights1, weights2, tie_breaking_rule)
            
        sec_exp = np.exp(alpha * sec_fitness)
        prob = sec_exp / (sec_exp + np.exp(alpha * prim_fitness))
            
        if np.random.binomial(1, prob):
            primary = secondary
        
    return primary


# project three-dimensional bids to coordinates on triangle
#@njit(nopython = True)
def bid_to_coordinate(bid, normalized = False):
    # ensure that bid has three dimensions
    assert len(bid) == 3
    if normalized:
        bid = bid / sum(bid)
    h = np.sqrt(3) / 2 
    coordinates = np.array([0, 0])
    coordinates = coordinates + bid[1] * np.array([1.0, 0.0])
    coordinates = coordinates + bid[2] * np.array([0.5, h])
    return(coordinates)

#@njit(nopython = True)
def plot_strategies(strategies, scores, symmetric_battlefields = True, labels = None):
    # initialize x and y
    if symmetric_battlefields:
        x = np.array([])
        y = np.array([])
        extended_scores = np.array([])
        extended_colors = np.array([])
        
        # apply k-means over the scores
        if labels is not None:
            colors = np.where(labels == 1, "green", "red")
        else:
            centroids, labels = kmeans2(scores, 2, iter = 100, minit = "points")
            colors = np.where(labels == np.argmax(centroids), "green", "red")
        
        n_battlefields = strategies.shape[1]
        n_perms_max = ss.perm(n_battlefields, n_battlefields, True)
        # compute the coordinates for each permutation
        for i in range(len(scores)):
            # do case distinguishing
            strategy_perms = get_unique_permutations(strategies[i], n_perms_max)
            n_perms = strategy_perms.shape[0]
            for j in range(n_perms):
                new_x, new_y = bid_to_coordinate(strategy_perms[j])
                x = np.append(x, new_x)
                y = np.append(y, new_y)
                extended_scores = np.append(extended_scores, scores[i])
                extended_colors = np.append(extended_colors, colors[i])
        
        # compute sizes from score values
        sizes = 10 + (extended_scores * strategies.max() / 2)
        return x, y, sizes, extended_colors
            
    else:
        x = np.zeros(len(scores))
        y = np.zeros(len(scores))
        
        for i in range(len(scores)):
            strategy = strategies[i]
            x[i], y[i] = bid_to_coordinate(strategy)
        
        sizes = 10 + (scores * strategies.max() / 2)
        if labels is not None:
            colors = np.where(labels == 1, "green", "red")
        else:
            centroids, labels = kmeans2(scores, 2, iter = 100, minit = "points")
            colors = np.where(labels == np.argmax(centroids), "green", "red")
    
    # return reversed so that green is visible
    return x, y, sizes, colors

# summarize the output by normalizing score values and average over symmetric strategies  
#@njit(nopython = True)
def summarize_output(count_strat1, count_strat2, strategies1, strategies2, symmetric_strategies, ordered_output):
    # compute the strategy space sizes
    # if there is only one strategies input assume symmetricity
    n_strategies1 = strategies1.shape[0]
    if strategies2 is None:
        strategies2 = strategies1
    n_strategies2 = strategies2.shape[0]
    
    count_strat1_sum = sum(count_strat1[:,1])
    count_strat2_sum = sum(count_strat2[:,1])

    if count_strat1_sum > 0:
        count_strat1[:,1] = count_strat1[:,1] / count_strat1_sum
    else: 
        # exception handling if there has never been a monomorphic state 
        warnings.warn('No monomorphic states for strategy 1 detected - consider decreasing the mutation rate or the population size')
        count_strat1[:,1] = np.repeat(1 / n_strategies1, n_strategies1)

    if count_strat2_sum > 0:
        count_strat2[:,1] = count_strat2[:,1] / count_strat2_sum 
    else:
        warnings.warn('No monomorphic states for strategy 2 detected - consider decreasing the mutation rate or the population size')
        count_strat2[:,1] = np.repeat(1 / n_strategies2, n_strategies2)

    # format ids
    count_strat1[:,0] = count_strat1[:,0].astype('int')  
    count_strat2[:,0] = count_strat2[:,0].astype('int')      
    
    if symmetric_strategies:
        count_strat_mean =  (count_strat1 * count_strat1_sum + count_strat2 * count_strat2_sum ) / (count_strat1_sum + count_strat2_sum )
        count_strat_mean[:,0] = count_strat_mean[:,0].astype('int')
        
        if ordered_output:
            ordering = np.argsort(count_strat_mean[:,1])[::-1]
            return [strategies1[ordering], count_strat_mean[ordering]]
        else:
            return [strategies1, count_strat_mean]
    else:
        if ordered_output:
            ordering1 = np.argsort(count_strat1[:,1])[::-1]
            ordering2 = np.argsort(count_strat2[:,1])[::-1]
            return [[strategies1[ordering1], count_strat1[ordering1]], [strategies2[ordering2], count_strat2[ordering2]]]
        else:
            return [[strategies1, count_strat1], [strategies2, count_strat2]]

# find all valid candidates for best responses
def best_response_candidates(strategies, symmetric_battlefields = True, less_budget = 0):
    n_battlefields = strategies.shape[1]
    # initialize with first iteration
    perturbation = np.ones(n_battlefields)
    perturbation[0] -= (n_battlefields + less_budget)
    best_response_candidates = strategies + perturbation
    # compute all remaining possible best responses
    for i in range(1, n_battlefields):
        perturbation = np.ones(n_battlefields)
        perturbation[i] -= (n_battlefields + less_budget)
        best_response_candidates = np.vstack((best_response_candidates, strategies + perturbation))
    # exclude invalid candidates with bids < 0
    valid_perturbations = np.sum(best_response_candidates < 0, axis = 1) == 0
    best_response_candidates = best_response_candidates[valid_perturbations]
    # order strategies if battlefields are symmetric
    if symmetric_battlefields:
        for i, cand in enumerate(best_response_candidates):
            best_response_candidates[i] = np.sort(cand)
    # only output unique strategies
    best_response_candidates = np.unique(best_response_candidates.astype(float), axis = 0)
    return best_response_candidates
        
# evaluate strategy set against its best response
def evaluate_strategy_subset(strategies, subset, weights1, weights2, budget, tie_breaking_rule, return_best_response = False, learnable_return = False):
    # check if battlefields are symmetric
    symmetric_battlefields = False
    if all(weights1 == weights2) and all(weights1 == 1):
        symmetric_battlefields = True
    
    n_battlefields = strategies.shape[1]
    # expand strategies
    if symmetric_battlefields:
        mixed_strategy = np.repeat(None, n_battlefields)
        n_perms = ss.perm(n_battlefields, n_battlefields, True)
        for strategy in subset:
            mixed_strategy = np.vstack((mixed_strategy, get_unique_permutations(strategy, n_perms)))
        # delete first line which was necessary for initializing
        mixed_strategy = mixed_strategy[1:]
    else:
        mixed_strategy = subset
    # remove duplicates
    mixed_strategy = np.unique(mixed_strategy.astype(float), axis = 0)
    support_size = mixed_strategy.shape[0]
    # get all the candidates for best response
    best_response_cand = best_response_candidates(strategies, symmetric_battlefields = symmetric_battlefields, less_budget = budget - sum(mixed_strategy[0]))
    # find the worst case fitness of the strategy set
    worst_case_loss = -1
    best_response = None
    for cand in best_response_cand:
        # now check the performance against the best response
        loss = 0
        for bid in mixed_strategy:
            loss += blotto_mechanism(cand, bid, weights2, weights1, tie_breaking_rule, learnable_return = learnable_return)
        # compare with current worst loss
        cand_loss = loss / support_size
        if cand_loss > worst_case_loss:
            worst_case_loss = cand_loss
            best_response = cand
        
    # compute and return average loss
    if return_best_response:
        return worst_case_loss, best_response
    return worst_case_loss

def find_best_mixed_strategy(strategies, scores, weights1, weights2, budget, tie_breaking_rule, mode = "kmeans"):
    if mode == "kmeans":
        centroids, labels = kmeans2(scores, 2, iter = 100, minit = "points")
        current_clusters = np.where(labels == np.argmax(centroids), True, False)
        evolutionary_stable = strategies[current_clusters]
        # evaluate by examining the order of the scores
        best_loss = evaluate_strategy_subset(strategies, evolutionary_stable, 
                                             weights1, weights2, budget, tie_breaking_rule)
        best_labels = current_clusters.astype(int)
    elif mode == "extensive":
        n_strategies = strategies.shape[0]
        labels = np.zeros(n_strategies)
        best_labels = None
        best_loss = math.inf
        # iterate through all splits according to the score values and select the best possible set of strategies
        for i in range(n_strategies):
            labels[i] = 1
            avg_loss = evaluate_strategy_subset(strategies, strategies[0:(i+1)], 
                                                weights1, weights2, budget, tie_breaking_rule)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_labels = labels.copy()
                
    return best_loss, best_labels

def evaluation_step(intermediate_result, symmetric_strategies, budget1, budget2, 
                    weights1, weights2, tie_breaking_rule, mode):
    if budget2 is None:
        budget2 = budget1
    if symmetric_strategies:
        # assert that input is consistent
        assert(budget1 == budget2)
        best_loss, best_labels = find_best_mixed_strategy(intermediate_result[0], intermediate_result[1][:,1], 
                                                          weights1, weights2, budget2, tie_breaking_rule, mode)
        return best_loss, best_labels
    else:     
        intermediate_result_p1 = intermediate_result[0]
        intermediate_result_p2 = intermediate_result[1]
        best_loss1, best_labels1 = find_best_mixed_strategy(intermediate_result_p1[0], intermediate_result_p1[1][:,1], 
                                                            weights1, weights2, budget2, tie_breaking_rule, mode)
        best_loss2, best_labels2 = find_best_mixed_strategy(intermediate_result_p2[0], intermediate_result_p2[1][:,1], 
                                                            weights2, weights1, budget1, tie_breaking_rule, mode)
        return [[best_loss1, best_loss2], [best_labels1, best_labels2]]
        
        
# top-level function to apply alpha_rank on two sets of strategies (one for each player in a two-player-game)          
#@njit(nopython = True)
def blotto_alpha_rank(strategies1, probs1, strategies2 = None, probs2 = None, weights1 = None, weights2 = None, tie_breaking_rule = "right-in-two",
                      pop_size = 10, alpha = 50, mr = 0.001, restarts = 10, epochs = 1000, 
                      ordered_output = True, track_every = 100, eval_mode = "kmeans", eval_every = 100, 
                      patience = 5, loss_goal = -1, plot_every = 100, surpress_plots = False):
    # if there is only one strategies input assume symmetricity
    n_strategies1 = strategies1.shape[0]
    if strategies2 is None:
        strategies2 = strategies1
        probs2 = probs1
    n_strategies2 = strategies2.shape[0]
    # compute budgets by assuming all strategies use it to its extent
    budget1 = np.sum(strategies1[0])
    budget2 = np.sum(strategies2[0])
    # setup weights if they are not set
    n_battlefields = strategies1.shape[1]
    if weights1 is None:
        weights1 = np.ones(n_battlefields)
    if weights2 is None:
        weights2 = weights1
    
    # check if battlefields are symmetric
    symmetric_battlefields = False
    if all(weights1 == weights2) and len(np.unique(weights1)) == 1:
        symmetric_battlefields = True
    # check if strategies are symmetric
    symmetric_strategies = False
    if (strategies1 == strategies2).all() and all(weights1 == weights2):
        symmetric_strategies = True
     
    # initialize counting arrays for ranking
    count_strat1 = np.zeros((n_strategies1, 2))
    count_strat1[:,0] = range(n_strategies1)
    count_strat2 = np.zeros((n_strategies2, 2))
    count_strat2[:,0] = range(n_strategies2)
    
    
    best_loss = 1
    best_mixed_strategy = None
    best_labels = None
    for restart in range(restarts):
        #initialize population
        pop1 = np.repeat(np.random.choice(n_strategies1), pop_size)
        pop2 = np.repeat(np.random.choice(n_strategies2), pop_size)
        
        remaining_patience = patience
    
        for i in range(epochs):
            # sample two distinct random strategies from the population for both players
            primary1_id = np.random.choice(pop_size)
            secondary1_id = primary1_id
            while secondary1_id == primary1_id:
                secondary1_id = np.random.choice(pop_size)
                
            primary2_id = np.random.choice(pop_size)
            secondary2_id = primary2_id
            while secondary2_id == primary2_id:
                secondary2_id = np.random.choice(pop_size)
                
            # get indices in the strategies variables
            primary1 = pop1[primary1_id]
            secondary1 = pop1[secondary1_id]
            primary2 = pop2[primary2_id]
            secondary2 = pop2[secondary2_id]
            
            # compute new playing strategy according to mutation or selection
            new_prim1 = selection_process(strategies1, probs1, primary1, secondary1, strategies2[primary2, :], 
                                          weights1 = weights1, weights2 = weights2, tie_breaking_rule = tie_breaking_rule, 
                                          alpha = alpha, mr = mr)
            new_prim2 = selection_process(strategies2, probs2, primary2, secondary2, strategies1[primary1, :], 
                                          weights1 = weights2, weights2 = weights1, tie_breaking_rule = tie_breaking_rule,
                                          alpha = alpha, mr = mr)
            
            pop1[primary1_id] = new_prim1
            pop2[primary2_id] = new_prim2
            
            # safe new population counts for monomorphic populations
            if len(np.unique(pop1)) == 1:
                count_strat1[pop1[0], 1] += 1
            
            if len(np.unique(pop2)) == 1:
                count_strat2[pop2[0], 1] += 1
            
            # count newly selected
            #count_strat1[new_prim1, 1] += 1
            #count_strat2[new_prim2, 1] += 1
            if (i + 1) % track_every == 0:
                print(restart * epochs + i + 1," out of ", epochs * restarts, " iterations done")
                print("Current best loss: " + str(best_loss))

            if (i + 1) % eval_every == 0:
                intermediate_result = summarize_output(count_strat1, count_strat2, strategies1, strategies2, symmetric_strategies, ordered_output = ordered_output)
                avg_loss, labels = evaluation_step(intermediate_result, symmetric_strategies, budget1, budget2,
                                                   weights1, weights2, tie_breaking_rule, eval_mode)
                # if there are multiple loss values, optimize their average
                if not symmetric_strategies:
                    avg_loss = np.mean(avg_loss)
                if avg_loss >= best_loss:
                    remaining_patience -= 1
                else:
                    best_loss = avg_loss
                    print("Current best loss :", best_loss)
                    best_labels = labels.copy()
                    best_mixed_strategy = intermediate_result
                    if not surpress_plots:
                        if symmetric_strategies:
                            x, y, sizes, colors = plot_strategies(best_mixed_strategy[0], best_mixed_strategy[1][:,1], symmetric_battlefields, best_labels)
                            plt.scatter(x, y, s = sizes, color = colors, alpha = 0.5 * (colors == "green").astype(int) + 0.5)
                            # get current axes
                            ax = plt.gca()
                            # hide y-axis because it is not accurate in this kind of plot
                            ax.get_yaxis().set_visible(False)
                            # show plot
                            plt.show()
                        else:
                            # plot for first player
                            x, y, sizes, colors = plot_strategies(best_mixed_strategy[0][0], best_mixed_strategy[0][1][:,1], symmetric_battlefields, best_labels[0])
                            plt.scatter(x, y, s = sizes, color = colors, alpha = 0.5 * (colors == "green").astype(int) + 0.5)
                            # get current axes
                            ax = plt.gca()
                            # hide y-axis because it is not accurate in this kind of plot
                            ax.get_yaxis().set_visible(False)
                            # show plot
                            plt.show()
                            # plot for second player
                            x, y, sizes, colors = plot_strategies(best_mixed_strategy[1][0], best_mixed_strategy[1][1][:,1], symmetric_battlefields, best_labels[1])
                            plt.scatter(x, y, s = sizes, color = colors, alpha = 0.5 * (colors == "green").astype(int) + 0.5)
                            # get current axes
                            ax = plt.gca()
                            # hide y-axis because it is not accurate in this kind of plot
                            ax.get_yaxis().set_visible(False)
                            # show plot
                            plt.show()
                    remaining_patience = patience
                
                if best_loss <= loss_goal:
                    remaining_patience = 0
                
                if remaining_patience == 0:
                    if symmetric_strategies:
                        print("Converged after ", restart * epochs + i + 1, " iterations with loss ", best_loss)
                        x, y, sizes, colors = plot_strategies(best_mixed_strategy[0], best_mixed_strategy[1][:,1], symmetric_battlefields, best_labels)
                        plt.scatter(x, y, s = sizes, color = colors, alpha = 0.5 * (colors == "green").astype(int) + 0.5)
                        # get current axes
                        ax = plt.gca()
                        # hide y-axis because it is not accurate in this kind of plot
                        ax.get_yaxis().set_visible(False)
                        # show plot
                        plt.show()
                    else:
                        # plot for first player
                        x, y, sizes, colors = plot_strategies(best_mixed_strategy[0][0], best_mixed_strategy[0][1][:,1], symmetric_battlefields, best_labels[0])
                        plt.scatter(x, y, s = sizes, color = colors, alpha = 0.5 * (colors == "green").astype(int) + 0.5)
                        # get current axes
                        ax = plt.gca()
                        # hide y-axis because it is not accurate in this kind of plot
                        ax.get_yaxis().set_visible(False)
                        # show plot
                        plt.show()
                        # plot for second player
                        x, y, sizes, colors = plot_strategies(best_mixed_strategy[1][0], best_mixed_strategy[1][1][:,1], symmetric_battlefields, best_labels[1])
                        plt.scatter(x, y, s = sizes, color = colors, alpha = 0.5 * (colors == "green").astype(int) + 0.5)
                        # get current axes
                        ax = plt.gca()
                        # hide y-axis because it is not accurate in this kind of plot
                        ax.get_yaxis().set_visible(False)
                        # show plot
                        plt.show()
                    return best_mixed_strategy, best_labels
            
            if ((i + 1) % round(plot_every) == 0):
                if symmetric_strategies:
                    intermediate_result = summarize_output(count_strat1, count_strat2, strategies1, strategies2, symmetric_strategies, ordered_output = ordered_output)
                    x, y, sizes, colors = plot_strategies(intermediate_result[0], intermediate_result[1][:,1], symmetric_battlefields)
                    plt.scatter(x, y, s = sizes, color = colors, alpha = 0.5 * (colors == "green").astype(int) + 0.5)
                    # get current axes
                    ax = plt.gca()
                    # hide y-axis because it is not accurate in this kind of plot
                    ax.get_yaxis().set_visible(False)
                    # show plot
                    plt.show()
                else:
                    # plot for first player
                    x, y, sizes, colors = plot_strategies(intermediate_result[0][0], intermediate_result[0][1][:,1], symmetric_battlefields)
                    plt.scatter(x, y, s = sizes, color = colors, alpha = 0.5 * (colors == "green").astype(int) + 0.5)
                    # get current axes
                    ax = plt.gca()
                    # hide y-axis because it is not accurate in this kind of plot
                    ax.get_yaxis().set_visible(False)
                    # show plot
                    plt.show()
                    # plot for second player
                    x, y, sizes, colors = plot_strategies(intermediate_result[1][0], intermediate_result[1][1][:,1], symmetric_battlefields)
                    plt.scatter(x, y, s = sizes, color = colors, alpha = 0.5 * (colors == "green").astype(int) + 0.5)
                    # get current axes
                    ax = plt.gca()
                    # hide y-axis because it is not accurate in this kind of plot
                    ax.get_yaxis().set_visible(False)
                    # show plot
                    plt.show()
                
        print(restart + 1, " out of ", restarts, " restarts done")
        
    # return the best mixed strategy and the labels of the actions in the support
    return best_mixed_strategy, best_labels
    