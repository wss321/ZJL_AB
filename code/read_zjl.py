# This code reads zjl-230 dataset and picks the first 5
# Classes for each superclass to enter the training procedure (target data)
# The other two labels will be used for zero-shot learning.

import pickle
import random
from sklearn.preprocessing import LabelBinarizer
from config import SPLIT_RATE, PKL_DIR, SAVE_VEC_PKL, SAVE_NTTRD_PKL, SAVE_TTRD_PKL, \
    SAVE_ALL_LABEL_PKL
from create_pickle_file import AB_META, AB_TRAIN_PICKLE

random.seed(0)


def read_pickle_file(filename):
    """Reads a pickle file using the latin1 encoding"""
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        # u.encoding = 'latin1'
        p = u.load()
    return p


def separate_target_data(zjl_dict, used_labels):
    """Separate classes to be used in zero shot inference using the ZJL-230  superclasses"""
    # FORMAT: DATA, FINE_LABEL, COARSE_LABEL
    target_data = []
    not_target_data = []

    len_data = len(zjl_dict['data'])

    for i in range(len_data):
        new_entry = [zjl_dict['data'][i], zjl_dict['fine_labels'][i],
                     zjl_dict['coarse_labels'][i]]

        fine = zjl_dict['fine_labels'][i]
        if fine in used_labels:
            target_data.append(new_entry)
        else:
            not_target_data.append(new_entry)

    return {'target': target_data, 'not_target': not_target_data}


def create_dataset_with_string_labels(dataset, metadata_dic):
    """Create a (image, fine_label, coarse_label) dataset"""
    new_dataset = []
    for d in dataset:
        fine_label = d[1]
        coarse_label = d[2]
        fine_name = metadata_dic['fine_label_names'][fine_label]
        coarse_name = metadata_dic['coarse_label_names'][coarse_label]
        new_dataset.append([d[0], fine_name, coarse_name])
    return new_dataset


def build_coarse_to_fine_correspondence(cifar_dict):
    """Create fine labels groups by superclass"""
    num_coarse = len(set(cifar_dict['coarse_labels']))
    corrs_coarse_fine = []
    for i in range(num_coarse):
        corrs_coarse_fine.append([])

    len_dict = len(cifar_dict['data'])

    for i in range(len_dict):
        coarse = cifar_dict['coarse_labels'][i]
        fine = cifar_dict['fine_labels'][i]
        if fine not in corrs_coarse_fine[coarse]:
            corrs_coarse_fine[coarse].append(fine)

    return corrs_coarse_fine


def separated_used_labels(coarse_to_fine_correspondence, tar_rate=SPLIT_RATE):
    """Within a superclass, separate zero shot and training labels"""
    used_labels = []
    all_labels = []

    for fine_labels in coarse_to_fine_correspondence:
        fine_num = len(fine_labels)
        tar_num = int(fine_num * tar_rate)
        used_labels += fine_labels[:tar_num]  # Pick the first three classes for target
        all_labels += fine_labels

    return [all_labels, used_labels]


# READING CIFAR 100 DATA

def read_zjl_main():
    # Create datasets
    print('Reading Data...')
    zjl230_train_dict = read_pickle_file(AB_TRAIN_PICKLE)  # TRAIN_PKL
    zjl230_meta = read_pickle_file(AB_META)  # META_PKL
    print(zjl230_meta['fine_label_names'])
    print(len(zjl230_train_dict['fine_labels']))
    print('Date read')

    print('CALCULATING CORRESPONDENCE')

    corrs_coarse_fine = build_coarse_to_fine_correspondence(zjl230_train_dict)
    [all_labels, used_labels] = separated_used_labels(corrs_coarse_fine, SPLIT_RATE)
    # 名
    used_labels_str = [zjl230_meta['fine_label_names'][L] for L in used_labels]
    print(used_labels_str)
    all_labels_str = [zjl230_meta['fine_label_names'][L] for L in all_labels]
    print('USED LABELS %d:' % len(used_labels_str), set(used_labels_str))
    print('ALL LABELS %d' % len(all_labels_str), set(all_labels_str))

    print('CORRESPONDENCE DONE')

    separated_train_data = separate_target_data(zjl230_train_dict, used_labels)
    zjl230_train_dict = None
    target_train_data = separated_train_data['target']
    not_target_train_data = separated_train_data['not_target']
    vectorizer = LabelBinarizer()
    vectorizer.fit(used_labels_str)
    # print(len(used_labels_str),used_labels_str)

    print('BUILDING DATASET WITH STR LABELS FOR NEW NORMALIZATION')
    str_target_train_data = create_dataset_with_string_labels(target_train_data, zjl230_meta)
    str_not_target_train_data = create_dataset_with_string_labels(not_target_train_data, zjl230_meta)
    print('SAVING TO {}'.format(PKL_DIR))

    out_target_train = open(SAVE_TTRD_PKL, 'wb')

    out_not_target_train = open(SAVE_NTTRD_PKL, 'wb')

    out_all_labels = open(SAVE_ALL_LABEL_PKL, 'wb')

    out_vectorizer = open(SAVE_VEC_PKL, 'wb')

    pickle.dump(str_target_train_data, out_target_train)
    str_target_train_data = None
    print(SAVE_TTRD_PKL, ' Done.')

    pickle.dump(str_not_target_train_data, out_not_target_train)
    str_not_target_train_data = None
    print(SAVE_NTTRD_PKL, ' Done.')

    pickle.dump(vectorizer, out_vectorizer)
    vectorize = None
    print(SAVE_VEC_PKL, ' Done.')
    pickle.dump(all_labels_str, out_all_labels)
    all_labels_str = None
    print(SAVE_ALL_LABEL_PKL, ' Done.')

    out_target_train.close()
    out_not_target_train.close()
    out_vectorizer.close()
    out_all_labels.close()

    print('ALL DONE!')


if __name__ == '__main__':
    read_zjl_main()
