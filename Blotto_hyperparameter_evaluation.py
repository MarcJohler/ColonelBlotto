import sys
import json
import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

r = open(sys.path[0] + "/outputs/output_no_noise_granularity15test.txt", "r")
scores = r.read()
r.close()

# replace unnecessary string parts
scores = scores.replace("\n" , "")
scores = scores.replace("array", "np.array")

expr = "score_dict = " + scores
exec(expr)

parameter_evaluation = {}
for key in score_dict.keys():
    entry = score_dict[key]
    strategies = entry[0]
    scores = entry[1][:,1]
    ne_ids = np.sum(strategies > 666, axis = 1) == 0
    non_ne_ids = ne_ids == False
    
    centroids, labels = kmeans2(scores, 2, iter = 100, minit = "points")
    classes = np.where(labels == np.argmax(centroids), True, False)
    
    # compute metrics 
    sensitivity = sum(ne_ids * classes) / sum(ne_ids)
    specificity = sum(non_ne_ids * (classes == False)) / sum(non_ne_ids)
    
    ch_score = calinski_harabasz_score(scores.reshape(-1, 1), classes.reshape(-1, 1))
    db_score = davies_bouldin_score(scores.reshape(-1, 1), classes.reshape(-1, 1))
    
    parameter_evaluation[key] = [sensitivity, specificity, ch_score, db_score]

for key in parameter_evaluation.keys(): 
    print(str(key) + ": " + str(parameter_evaluation[key]))