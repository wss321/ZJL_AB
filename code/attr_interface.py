import numpy as np
import pickle
from config import attr_data_file
from normalize import attr_normal_from_file
from create_pickle_file import AB_META
np.random.seed(0)
print('Loading attributes')
meta = pickle.load(open(AB_META, 'rb'))
attrs = attr_normal_from_file(attr_data_file).set_index(0)
# print(attrs)
print('Loaded')

norm_mean = 1  # 5.5293


def word_to_cid(word):
    class2cid = meta['class2cid']
    try:
        return class2cid[word]
    except:
        return None


def find_norm_mean():
    """Find the mean norm of the attributes representations"""
    all_attrs = attrs.index.values
    count = .0
    norm_sum = .0

    for a in all_attrs:
        new_norm = np.linalg.norm(attrs.loc[a].as_matrix())
        norm_sum += new_norm
        count += 1
    norm_sum /= count
    return norm_sum


def find_attr_vec(word):
    """Gets the attribute representation from a word"""
    try:
        cid = word_to_cid(word)
        return attrs.loc[cid].as_matrix() / norm_mean
    except:
        return None
