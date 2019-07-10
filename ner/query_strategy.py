# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 12:13:12 2017
The code are functions for different query strategies for active learning:
    random sampling, uncertainty(maxmium entropy) sampling, diversity sampling
@author: lming
"""
import random

import numpy as np
import math


def randomKSamples(x_un, y_un, num):
    # x_un=np.array(x_un)
    # y_un=np.array(y_un)
    querydata = []
    querylabels = []
    queryindices = []
    x_new = []
    y_new = []
    indices = np.arange(len(x_un))
    np.random.shuffle(indices)
    for i in range(0, num):
        querydata.append(x_un[indices[i]])
        querylabels.append(y_un[indices[i]])
        queryindices.append(indices[i])
    return querydata, querylabels, queryindices


# randomly sample num examples from x_un and y_un, also return the new x_un, y_un
def randomSample(x_un, y_un, num):
    data = list(zip(x_un, y_un))
    random.shuffle(data)
    sample_idxs = random.sample(range(len(data)), num)
    samples = [data[i] for i in sample_idxs]
    new_data = []
    for i in range(len(data)):
        if i not in sample_idxs:
            new_data.append(data[i])
    querydata, querylabels = [list(t) for t in zip(*samples)]
    x_new, y_new = [list(t) for t in zip(*new_data)]
    return querydata, querylabels, x_new, y_new

# uncertainty (max predictive entropy) based sampling
def uncertaintySample(x_un, y_un, num, model, k=5):
    x_un = np.array(x_un)
    y_un = np.array(y_un)
    querydata = []
    querylabels = []
    entropy = []
    x_new = []
    y_new = []
    entropy = [model.get_uncertainty(item) for item in x_un]
    # y_predict = [model.get_predictions(item) for item in x_un]
    # for prediction in y_predict:
    #     entropy.append(getEntropy(prediction))
    indices = sorted(range(len(entropy)), key=lambda i: entropy[i], reverse=True)
    for i in range(0, num):
        querydata.append(x_un[indices[i]])
        querylabels.append(y_un[indices[i]])
    for i in range(num, len(x_un)):
        x_new.append(x_un[indices[i]])
        y_new.append(y_un[indices[i]])
    return querydata, querylabels, x_new, y_new


def getuncertaintyRank(x_un, x_singalnew, model):
    x_un = np.array(x_un)
    rank = 1
    entropy = []
    x_new = np.expand_dims(x_singalnew, axis=0)
    y_new = model.predict(x_new)[0]
    y_new_entropy = getEntropy(y_new)
    y_predict = model.predict(x_un)
    for prediction in y_predict:
        entropy.append(getEntropy(prediction))
    sorted_entropy = sorted(entropy, reverse=True)

    for i in range(0, len(sorted_entropy)):
        if ((y_new_entropy - sorted_entropy[i]) > 0):
            rank = i + 1
            break
        else:
            continue
    return rank


def diversitySample(x_un, y_un, num, x_train, k=5):
    x_un = np.array(x_un)
    y_un = np.array(y_un)
    querydata = []
    querylabels = []
    x_new = []
    y_new = []
    diversity = []
    results_union = set().union(*x_train)
    for example in x_un:
        value = jaccard_similarity(example, results_union)
        diversity.append(value)
    indices = sorted(range(len(diversity)), key=lambda i: diversity[i])
    for i in range(0, num):
        querydata.append(x_un[indices[i]])
        querylabels.append(y_un[indices[i]])
    for i in range(num, len(x_un)):
        x_new.append(x_un[indices[i]])
        y_new.append(y_un[indices[i]])
    return querydata, querylabels, x_new, y_new


def getdiversityRank(x_un, x_singalnew, x_train):
    rank = 1
    x_un = np.array(x_un)
    diversity = []
    results_union = set().union(*x_train)
    y_new_similarity = jaccard_similarity(x_singalnew, results_union)
    for example in x_un:
        value = jaccard_similarity(example, results_union)
        diversity.append(value)
    sorted_diversity = sorted(diversity, reverse=True)
    for i in range(0, len(sorted_diversity)):
        if ((y_new_similarity - sorted_diversity[i]) > 0):
            rank = i + 1
            break
        else:
            continue
    return rank


# diversity-based sampling
def diversityallSample(x_un, y_un, num, x_train):
    x_un = np.array(x_un)
    y_un = np.array(y_un)
    querydata = []
    querylabels = []
    x_new = []
    y_new = []
    diversity = []
    for example in x_un:
        n = 0
        value = 0
        for point in x_train:
            value += jaccard_similarity(example, point)
            n = n + 1
        if n == 0:
            value = 0
        else:
            value = value / n
        diversity.append(value)
    indices = sorted(range(len(diversity)), key=lambda i: diversity[i])
    for i in range(0, num):
        querydata.append(x_un[indices[i]])
        querylabels.append(y_un[indices[i]])
    for i in range(num, len(x_un)):
        x_new.append(x_un[indices[i]])
        y_new.append(y_un[indices[i]])
    return querydata, querylabels, x_new, y_new


# get the entropy of an vector (predictive distribution)
def getEntropy(v):
    entropies = [(-p) * np.log(p) for element in v for p in element]
    return np.sum(entropies)


def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / float(union_cardinality)


def square_rooted(x):
    return round(math.sqrt(sum([a * a for a in x])), 3)


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 5)


'''   
a=[[1,2,3,4],[1,2,0,4],[1,2,0,0],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
b=[[0,1],[0,1],[0,1],[01,1],[01,1],[01,1],[01,1]]

querydata, querylabels, x_new, y_new=randomSample(a, b, 6)
print(querydata)
print(querylabels)
print(x_new)
print(y_new)

c=[0.1,0.0,0.1, 0.1, 0.9, 0.3]
b=[0.3,0.2,0.2, 0.4, 0.1, 0.0]

print(cosine_similarity(b,c))
'''