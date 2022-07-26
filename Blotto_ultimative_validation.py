# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 17:40:05 2022

@author: marc
"""

from Blotto_alpha_rank import blotto_alpha_rank, evaluate_strategy_subset, blotto_mechanism, get_unique_permutations, plot_strategies
import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt

class Blotto_Defender:
    def __init__(self, strategies1, probs1, strategies2, probs2, weights1, weights2, tie_breaking_rule,
                 pop_size, alpha, mr, 
                 restarts, epochs, ordered_output, 
                 track_every, eval_mode, eval_every, patience, loss_goal, 
                 plot_every, surpress_plots):
        self.strategies1 = strategies1
        self.strategies2 = strategies2
        self.probs1 = probs1
        self.probs2 = probs2
        self.weights1 = weights1
        self.weights2 = weights2
        self.tie_breaking_rule = tie_breaking_rule
        self.pop_size = pop_size
        self.alpha = alpha
        self.mr = mr
        self.restarts = restarts
        self.epochs = epochs
        self.ordered_output = ordered_output 
        self.track_every = track_every
        self.eval_mode = eval_mode
        self.eval_every = eval_every
        self.patience = patience
        self.loss_goal = loss_goal
        self.plot_every = plot_every
        self.surpress_plots = surpress_plots
        # attributes generated by methods
        self.current_strategy = None
        self.n_battlefields = strategies1.shape[1]
        self.n_strategies = strategies1.shape[0]
        self.n_perms_max = ss.perm(self.n_battlefields, self.n_battlefields, True)
        self.action_frequencies = np.zeros(strategies1.shape[0])
        
    def generate_random_strategy(self):
        ranks, labels  = blotto_alpha_rank(self.strategies1, self.probs1, self.strategies2, self.probs2, self.weights1, self.weights2, self.tie_breaking_rule,
                                           pop_size = self.pop_size, alpha = self.alpha, mr = self.mr, 
                                           restarts = self.restarts, epochs = self.epochs, ordered_output = False, 
                                           track_every = self.track_every, eval_mode = self.eval_mode, eval_every = self.eval_every, patience = self.patience, loss_goal = self.loss_goal, 
                                           plot_every = self.plot_every, surpress_plots = self.surpress_plots)
        
        # remember labels of actions 
        self.labels = labels
        # extract chosen actions for support of mixed strategy
        unmirrored_strategies = ranks[0][labels == 1]
        mirrored_strategies = np.repeat(None, self.n_battlefields).reshape((1, self.n_battlefields))
        for i in range(unmirrored_strategies.shape[0]):
            # do case distinguishing
            strategy_perms = get_unique_permutations(unmirrored_strategies[i], self.n_perms_max)
            n_perms = strategy_perms.shape[0]
            for j in range(n_perms):
                mirrored_strategies = np.vstack((mirrored_strategies, strategy_perms[j]))
        # delete empty strategy
        mirrored_strategies = mirrored_strategies[1:]
        self.current_strategy = mirrored_strategies
        # print the support size of the current_strategy
        print("Support size of defender strategy: ", mirrored_strategies.shape[0], "\n")
    
    def sample_action(self, batch_size):
        support_size = self.current_strategy.shape[0]
        random_ids = np.random.choice(support_size, size = batch_size, replace = True)
        random_actions = self.current_strategy[random_ids]
        # save chosen action in history
        for i in range(batch_size):
            random_action = random_actions[i]
            ordered_random_action = np.sort(random_action)
            # lookup strategy in original strategy set
            is_equal = np.sum(self.strategies1 == ordered_random_action, axis = 1) == self.n_battlefields
            self.action_frequencies[is_equal] += 1
        
        return random_actions
        
    
class Blotto_Attacker:
    def __init__(self, strategies, budget, symmetric_battlefields, own_weights, opponent_weights, tie_breaking_rule, memory_size):
        self.strategies = strategies
        self.budget = budget
        self.symmetric_battlefields = symmetric_battlefields
        self.own_weights = own_weights
        self.opponent_weights = opponent_weights
        self.tie_breaking_rule = tie_breaking_rule
        self.memory_size = memory_size
        # remember bids of opponent
        self.opponent_history = None
        self.current_strategy = None
    def remember_strategy(self, strategy):
        # reshape single strategy
        if len(strategy.shape) == 1:
            strategy = strategy.reshape((1, len(strategy)))
        if self.opponent_history is None:
            self.opponent_history = strategy
        else:
            self.opponent_history = np.vstack((self.opponent_history, strategy))
        
        if self.opponent_history.shape[0] > self.memory_size:
            self.opponent_history = self.opponent_history[1:]
        
    def exploit_opponent(self):
        unique_history = np.unique(self.opponent_history.astype(float), axis = 0)
        
        loss, best_response = evaluate_strategy_subset(self.strategies, unique_history, 
                                                       self.opponent_weights, self.own_weights,
                                                       self.budget, self.tie_breaking_rule, return_best_response = True)
        self.current_strategy = best_response
        return best_response
        
        

def blotto_ultimative_validation(strategies1, probs1, strategies2 = None, probs2 = None, weights1 = None, weights2 = None, tie_breaking_rule = "right-in-two",
                                 symmetric_battlefields = True, pop_size = 10, alpha = 50, mr = 0.01, 
                                 restarts = 1, batch_size = 100, outer_epochs = 100, inner_epochs = 1000, 
                                 ordered_output = True, track_every = 100, eval_mode = "kmeans", eval_every = 100, 
                                 patience = 5, loss_goal = 0.12, plot_every = 100, surpress_plots = True):

    blotto_defender = Blotto_Defender(strategies1, probs1, strategies2, probs2, weights1, weights2, tie_breaking_rule,
                                      pop_size, alpha, mr, 
                                      restarts, inner_epochs, ordered_output, 
                                      track_every, eval_mode, eval_every, patience, loss_goal, 
                                      plot_every, surpress_plots)
    blotto_attacker = Blotto_Attacker(strategies1, strategies1[0].sum(), symmetric_battlefields, weights2, weights1, tie_breaking_rule, batch_size)
    # initialize defender with learned strategy
    blotto_defender.generate_random_strategy()
    # sample random action from defender strategy and initialize attacker strategy
    blotto_attacker.remember_strategy(blotto_defender.sample_action(1))
    # Let them battle
    avg_loss = 0
    avg_counter = 0
    all_time_loss = 0
    for j in range(outer_epochs):
        # generate a random defender strategy and sample action after a certain number of epochs
        blotto_defender.generate_random_strategy()
        defender_actions = blotto_defender.sample_action(batch_size)
        # compute loss
        loss_vec = np.zeros(batch_size)
        # let attacker attack defender and exploit the observed actions
        for i, action in enumerate(defender_actions):
            attacker_action = blotto_attacker.exploit_opponent()
            loss_vec[i] = blotto_mechanism(attacker_action, action)
            blotto_attacker.remember_strategy(action)
        loss = np.mean(loss_vec)
        # compute running average
        avg_loss = (avg_loss * avg_counter + loss) / (avg_counter + 1)
        avg_counter += 1
        all_time_loss = (all_time_loss * j + loss) / (j + 1)
        # print loss values
        if (j + 1) % 10 == 0:
            print(str(j + 1) + " out of " + str(outer_epochs) + " epochs done" + "\n")
            print("Current average loss for defender: ", avg_loss, "\n")
            print("All-time loss for defender: ", all_time_loss, "\n")
            avg_counter = 0
            avg_loss = 0
            # also plot the current historic distribution of actions played
            # normalize frequencies for usage in plots
            action_frequencies = blotto_defender.action_frequencies / sum(blotto_defender.action_frequencies)
            x, y, sizes, colors = plot_strategies(strategies1, action_frequencies, symmetric_battlefields, action_frequencies > 0)
            plt.scatter(x, y, s = sizes, color = colors, alpha = 0.5 * (colors == "green").astype(int) + 0.5)
            # get current axes
            ax = plt.gca()
            # hide y-axis because it is not accurate in this kind of plot
            ax.get_yaxis().set_visible(False)
            # show plot
            plt.show()
            
    return all_time_loss, blotto_defender.action_frequencies
            
        
