import sys
import json
import numpy as np
from Blotto_alpha_rank import evaluate_strategy_subset
from scipy.cluster.vq import kmeans2
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score


def evaluate_hyperparameters(rel_path, strategies, weights1, weights2, tie_breaking_rule):
    r = open(sys.path[0] + "/" + rel_path, "r")
    scores = r.read()
    r.close()
    
    # replace unnecessary string parts
    scores = scores.replace("\n" , "")
    scores = scores.replace("array", "numpy.array")
    
    d = {}
    expr = "score_dictionary = " + scores
    import_code = compile("import numpy", "import numpy", "exec")
    code = compile(expr, "assign scores", "exec")
    exec(import_code, d)
    exec(code, d)
    score_dictionary = d['score_dictionary']
    
    n_battlefields = strategies.shape[1]
    budget = sum(strategies[0])
    
    parameter_evaluation = {}
    for key in score_dictionary.keys():
        entry = score_dictionary[key]
        ranks = entry[0]
        labels = entry[1]
        strategies = ranks[0]
        mixed_strategy = strategies[labels == 1]
        scores = ranks[1][:,1]
        ne_ids = np.sum(strategies > 2 / n_battlefields * budget, axis = 1) == 0
        non_ne_ids = ne_ids == False
        
        centroids, labels = kmeans2(scores, 2, iter = 100, minit = "points")
        classes = np.where(labels == np.argmax(centroids), True, False)
        
        # compute metrics 
        loss = "mixed strategy is empty"
        if mixed_strategy.shape[0] != 0:
            loss = evaluate_strategy_subset(strategies, mixed_strategy, weights1, weights2, budget, tie_breaking_rule)
        
        sensitivity = sum(ne_ids * classes) / sum(ne_ids)
        specificity = sum(non_ne_ids * (classes == False)) / sum(non_ne_ids)
        
        ch_score = calinski_harabasz_score(scores.reshape(-1, 1), classes.reshape(-1, 1))
        db_score = davies_bouldin_score(scores.reshape(-1, 1), classes.reshape(-1, 1))
        
        parameter_evaluation[key] = [loss, sensitivity, specificity, ch_score, db_score]
    
    for key in parameter_evaluation.keys(): 
        print(str(key) + ": " + str(parameter_evaluation[key]))