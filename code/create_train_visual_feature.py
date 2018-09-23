from keras.applications.vgg19 import preprocess_input

import numpy as np
import tqdm
import os
import pandas as pd
from keras.preprocessing import image
import pickle
from config import SAVE_DIR, NUM_CHANNELS, DATAB_ALL_DIR, DATAB_TRAIN_DIR, DATAA_TRAIN_DIR, DATAA_ALL_DIR, DATASET
from keras import Model
from keras.models import load_model
from config import BEST_CLASSIFY_CKPT_FILE
from read_zjl import read_pickle_file

if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

np.random.seed(0)


def spilt_file(file_path):
    return_data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            return_data.append(line.split())
    return return_data


TRAIN_FEATURE_PATH = SAVE_DIR + '/train_{}_visual_features_and_fine_names_kaggle_dense.pkl'.format(DATASET)


def create_train_visual_feature_main():
    from config import VISUAL_SIZE, last_layer
    from create_pickle_file import TRAIN_PICKLE, IMAGE_SIZE, NUM_CHANNELS
    from dataset_utils import FILE_NAME_CID_PATH
    all_train_fine_labels = read_pickle_file(FILE_NAME_CID_PATH)['all_train_fine_labels']
    print('LOADING CLASSIFIER AT {} ...'.format(BEST_CLASSIFY_CKPT_FILE))
    model = load_model(BEST_CLASSIFY_CKPT_FILE)
    model = Model(inputs=model.input, outputs=model.get_layer(last_layer).output)
    print('LOADING TRAIN DATA....')
    X = read_pickle_file(TRAIN_PICKLE)['data']
    X = X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    print('Creating training visual feature .... ')
    features = model.predict(X, verbose=1).reshape(-1, VISUAL_SIZE)
    features_f = open(TRAIN_FEATURE_PATH, 'wb')
    pickle.dump({'features': features, 'fine_label_names': all_train_fine_labels}, features_f, protocol=4)
    features_f.close()
    print("Done.")


if __name__ == '__main__':
    create_train_visual_feature_main()
