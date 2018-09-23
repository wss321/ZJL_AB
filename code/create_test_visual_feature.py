# Code to create the visual feature of test image

import numpy as np
import os
import pickle
from config import SAVE_DIR
from keras import Model

from keras.models import load_model
from config import BEST_CLASSIFY_CKPT_FILE

if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
np.random.seed(0)

TEST_FEATURE_PATH = SAVE_DIR + '/test_visual_features_and_fine_names_kaggle_dense.pkl'


def spilt_file(file_path):
    return_data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            return_data.append(line.split())
    return return_data


def create_test_visual_feature_main():
    from config import VISUAL_SIZE, last_layer
    from create_pickle_file import TEST_PICKLE, IMAGE_SIZE, NUM_CHANNELS
    from read_zjl import read_pickle_file
    print('LOADING TEST DATA....')
    X = read_pickle_file(TEST_PICKLE).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    print('LOADING CLASSIFIER AT {} ...'.format(BEST_CLASSIFY_CKPT_FILE))
    model = load_model(BEST_CLASSIFY_CKPT_FILE)
    model = Model(inputs=model.input, outputs=model.get_layer(last_layer).output)
    print('Creating Test Feature...')
    features = model.predict(X, verbose=1).reshape(-1, VISUAL_SIZE)

    with open(TEST_FEATURE_PATH, 'wb') as features_f:
        pickle.dump({'features': features}, features_f, protocol=4)
    print("Done.")


if __name__ == '__main__':
    create_test_visual_feature_main()
