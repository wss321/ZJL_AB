from batch_making import not_target_train_data
import pickle
import numpy as np
from word2vec_interface import find_word_vec
from attr_interface import find_attr_vec
from config import ALL_LABLES_FILE, WV_OR_ATTR
np.random.seed(0)
not_target_labels = list(set([not_target_train_data[i][1] for i in range(len(not_target_train_data))]))
print('{} not used labels is:\n {}'.format(len(not_target_labels), not_target_labels))
all_names = pickle.load(open(ALL_LABLES_FILE, 'rb'))
print(all_names)


def cosine_distance(v1, v2):
    """Computes the cossine distance between two vectors"""
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_closest_words_eucl(vector, zero_shot_only=False):
    """Returns the closest word or attr to a vector in a crescent distance order.
    Uses euclidean distance"""
    all_distances = []
    possible_labels = all_names
    if zero_shot_only:
        possible_labels = not_target_labels

    for label in possible_labels:
        if WV_OR_ATTR == 'WV':
            fn = find_word_vec
        else:
            fn = find_attr_vec
        vec = fn(label)
        all_distances.append([label, np.linalg.norm(vector - vec)])
    sorted_dist = sorted(all_distances, key=lambda x: x[1])
    return [s[0] for s in sorted_dist]


def get_closest_words_cosine(vector, zero_shot_only=False):
    """Returns the closest words or attr to a vector in a crescent distance order.
    Uses cossine distance"""
    all_distances = []
    possible_labels = all_names
    if zero_shot_only:
        possible_labels = not_target_labels

    for label in possible_labels:
        if WV_OR_ATTR == 'WV':
            fn = find_word_vec
        else:
            fn = find_attr_vec
        vec = fn(label)
        all_distances.append([label, cosine_distance(vector, vec)])
    sorted_dist = sorted(all_distances, key=lambda x: x[1])
    return [s[0] for s in sorted_dist]
