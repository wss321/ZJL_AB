import pandas as pd
import numpy as np
import os
from config import DATAB_ALL_DIR

np.random.seed(0)


def normalline(data):
    for i, cs in enumerate(np.sum(data, axis=1).values):
        if cs != 0:
            data.iloc[i, :] = data.iloc[i, :] * 1.0 / cs
    return data


def attr_normal_from_file(src_dir, save_to=None):
    attr = pd.read_csv(src_dir, header=None, sep='\t')

    labels = np.array(attr.iloc[:, 0]).reshape(-1, 1)

    dentype = attr.iloc[:, 1:7]
    dentype = normalline(dentype)

    dencolor = attr.iloc[:, 7:15]
    dencolor = normalline(dencolor)

    denhas = attr.iloc[:, 15:19]
    denhas = normalline(denhas)

    denfor = attr.iloc[:, 19:25]
    denfor = normalline(denfor)

    attr = pd.DataFrame(np.concatenate((labels, dentype, dencolor, denhas, denfor), axis=1))
    if save_to:
        attr.to_csv(save_to, header=None, sep='\t', index=None)
    return attr


def word2vec_norm_from_file(src_dir, save_to=None):
    word2vec = pd.read_csv(src_dir, header=None, sep=' ')
    labels = np.array(word2vec.iloc[:, 0]).reshape(-1, 1)
    data = word2vec.iloc[:, 1:]
    data = normalline(data)
    word2vec = pd.DataFrame(np.concatenate((labels, data), axis=1))
    if save_to:
        word2vec.to_csv(save_to, header=None, sep='\t', index=None)
    return word2vec


if __name__ == '__main__':
    att = attr_normal_from_file(os.path.join(DATAB_ALL_DIR, 'attributes_per_class.txt'))
    print(att)
    from config import word2vec_data_file

    words = word2vec_norm_from_file(word2vec_data_file)
    print(words)
