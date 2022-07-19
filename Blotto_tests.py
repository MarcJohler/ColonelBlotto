from Blotto_alpha_rank import bid_to_coordinate, evaluate_strategy_set, blotto_alpha_rank
from Blotto_discretizer import discretize_action_space
import numpy as np
import statistics


strat, prob = discretize_action_space(3, 1500, symmetric_battlefields = True, granularity_level = 15, add_noise = False, integer_bids = True)

valid_strat_ids = (strat > 2/3 * 1500).sum(axis = 1) == 0
valid_strats = strat[valid_strat_ids]
#valid_strats = np.array([[300., 450., 750.], [150., 600., 750.]])

avg_loss = evaluate_strategy_set(valid_strats, True)

#print(valid_strats)
#print(avg_loss)

good_strategy = np.array([[  0., 428., 572.],
       [142., 429., 429.],
       [  0., 285., 715.],
       [142., 143., 715.],
       [ 90., 364., 546.],
       [272., 273., 455.],
       [181., 182., 637.],
       [  0., 454., 546.],
       [  0., 384., 616.],
       [ 76., 231., 693.],
       [153., 308., 539.]])


symmetric_battlefields = True
epochs = 10**6
mode = "extensive"
eval_every = 10**2
patience = 1000
mr = 0.1
pop_size = 10
restarts = 1
plot_every = 10**5

ranks, labels  = blotto_alpha_rank(good_strategy, "uniform", None, None, symmetric_battlefields = symmetric_battlefields, 
                          pop_size = pop_size, alpha = 100, mr = mr, 
                          restarts = restarts, epochs = epochs, 
                          track_every = plot_every, eval_mode = mode, eval_every = eval_every, patience = patience, plot_every = plot_every)

print(ranks)
print(labels)