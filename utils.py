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

def readGz(path):
    for l in gzip.open(path, 'rt', encoding = 'utf8'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

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

def naive_jaccard_model(dataset_folder_path):
    user_recommend = {}

    for  d in readGz(os.path.join(dataset_folder_path,"australian_user_reviews.json.gz")):
        user_recommend[d['user_id']] = [review["item_id"] for review in d['reviews'] if review["recommend"]==True]


    steam_games_genres = {}
    # collect all the genres, tags, specs info for each game

    steam_games_genres = {}

    missing_genres_count = 0
    missing_tags_count = 0
    missing_specs_count = 0
    missing_id_count = 0
    rep_items = 0

    for d in readGz(os.path.join(dataset_folder_path,"steam_games.json.gz")):

        if 'id' not in d.keys():
            print(f"This game has no id info!")
            print(d)
            missing_id_count += 1
            continue

        if d['id'] in steam_games_genres.keys():
            print(f"This game id ({d['id']}) already exists!")
            rep_items += 1
            continue

        if 'genres' not in d.keys():
            print(f"This game id ({d['id']}) has no genre info!")
            genre_info = []
            missing_genres_count += 1

        else:
            genre_info = d['genres']

        if 'tags' not in d.keys():
            print(f"This game id ({d['id']}) has no tag info!")
            tag_info = []
            missing_tags_count += 1

        else:
            tag_info = d['tags']

        if 'specs' not in d.keys():
            print(f"This game id ({d['id']}) has no specs info!")
            specs_info = []
            missing_specs_count += 1

        else:
            specs_info = d['specs']

        steam_games_genres[d['id']] = {'genres': genre_info,
                                    'tags': tag_info, 'specs': specs_info}

    # collect the genre, tags, specs info of each game in each bundle
    bundle_games_genres = {}

    for d in readGz(os.path.join(dataset_folder_path,"bundle_data.json.gz")):

        if "bundle_id" not in d.keys() or "items" not in d.keys():
            print(f"This bundle data is missing id or items info!")
            print(d)
            continue

        games_ids = [g["item_id"] for g in d["items"]]

        bundle_games_genres[d["bundle_id"]] = {'genres': [], 'tags': [], 'specs': []}
        for item in games_ids:
            if item in steam_games_genres.keys():
                bundle_games_genres[d["bundle_id"]]['genres'].extend(steam_games_genres[item]['genres'])
                bundle_games_genres[d["bundle_id"]]['tags'].extend(steam_games_genres[item]['tags'])
                bundle_games_genres[d["bundle_id"]]['specs'].extend(steam_games_genres[item]['specs'])

        bundle_games_genres[d["bundle_id"]]['genres'] = list(set(bundle_games_genres[d["bundle_id"]]['genres']))
        bundle_games_genres[d["bundle_id"]]['tags'] = list(set(bundle_games_genres[d["bundle_id"]]['tags']))
        bundle_games_genres[d["bundle_id"]]['specs'] = list(set(bundle_games_genres[d["bundle_id"]]['specs']))

    # for each user, collect all the genres, tags, specs info from the games they recommended

    user_recommend_genres = {}
    for user in user_recommend.keys():
        user_recommend_genres[user] = {'genres': [], 'tags': [], 'specs': []}
        for item in user_recommend[user]:
            if item in steam_games_genres.keys():
                user_recommend_genres[user]['genres'].extend(steam_games_genres[item]['genres'])
                user_recommend_genres[user]['tags'].extend(steam_games_genres[item]['tags'])
                user_recommend_genres[user]['specs'].extend(steam_games_genres[item]['specs'])

    # find jaccard similarity between games in a bundle and user's preferred genres/tags/specs
    # let teh threshold be 0.5 for now

    total_len = len(user_recommend_genres.keys())

    jaccard_similarity_threshold = 0.8
    user_bundle_match = defaultdict(list)
    for user in user_recommend_genres.keys():

        print(f"Processing user {user} ({list(user_recommend_genres.keys()).index(user)+1}/{total_len})")
        user_genres_set = set(user_recommend_genres[user]['genres'])
        user_tags_set = set(user_recommend_genres[user]['tags'])
        user_specs_set = set(user_recommend_genres[user]['specs'])

        for bundle in bundle_games_genres.keys():
            bundle_genres_set = set(bundle_games_genres[bundle]['genres'])
            bundle_tags_set = set(bundle_games_genres[bundle]['tags'])
            bundle_specs_set = set(bundle_games_genres[bundle]['specs'])

            genre_jaccard = jaccard_similarity(user_genres_set, bundle_genres_set)
            tag_jaccard = jaccard_similarity(user_tags_set, bundle_tags_set)
            specs_jaccard = jaccard_similarity(user_specs_set, bundle_specs_set)

            if (np.mean([genre_jaccard, tag_jaccard, specs_jaccard])
                >= jaccard_similarity_threshold):
                user_bundle_match[user].append(bundle)

    data_cleaning_stats = (missing_genres_count, missing_tags_count,
                              missing_specs_count, missing_id_count, rep_items)

    return user_bundle_match, data_cleaning_stats

def evaluation_function(test_tuple_list, user_bundle_match):

    result_list = []
    list_len = len(test_tuple_list)

    for d in test_tuple_list:

        print(f"Evaluating test case {test_tuple_list.index(d)+1}/{list_len}")

        user_id = d[0]
        pos_bundle = d[1]
        neg_bundles = d[2]

        if user_id in user_bundle_match.keys():
            recommended_bundles = user_bundle_match[user_id]
        else:
            recommended_bundles = []

        if (pos_bundle in recommended_bundles) and (neg_bundles not in recommended_bundles):
            result_list.append(1)
        else:
            result_list.append(0)

    return np.mean(result_list)

def evaluation_function_naive(test_tuple_list, user_bundle_match = None):

    if user_bundle_match is None:
        try:
            user_bundle_match, _ = naive_jaccard_model(os.path.join(os.getcwd(),"dataset"))
        except FileNotFoundError:
            print("Dataset folder not found. Please provide user_bundle_match dictionary.")
            return None

    res_acc = evaluation_function(test_tuple_list, user_bundle_match)

    return res_acc