from collections import defaultdict
import math
import scipy.optimize
import numpy
import string
from sklearn import linear_model
import random
import gzip
import numpy as np
from collections import Counter
import os

def auc(pos_scores, neg_scores):
    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    # Create all pairwise comparisons
    diff = pos_scores[:, None] - neg_scores[None, :]

    # Count how often positive scores are higher
    return np.mean(diff > 0)

def jaccard_similarity(list1, list2):
    # use with list of strings or numbers
    set1 = set(list1)
    set2 = set(list2)

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if not union:
        return 0.0   # avoid division by zero

    return len(intersection) / len(union)

def cosine_similarity(vec1, vec2):
    # use with list or numpy arrays of numbers
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0   # avoid division by zero

    return dot_product / (norm1 * norm2)