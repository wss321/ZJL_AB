# Code to create the visual feature of test image
from keras.applications.vgg19 import preprocess_input

import numpy as np
import tqdm
import os
import pandas as pd
from keras.preprocessing import image
import pickle
from config import SAVE_DIR, DATAB_ALL_DIR, TEST_DATA_DIR
from keras import Model

from keras.models import load_model
from keras_train import BAET_CLASSIFY_CKPT_FILE

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
    from config import KERAS_MODEL as model
    if model == 'densenet':
        last_layer = 'global_average_pooling2d_1'
    else:
        last_layer = 'batch_normalization_19'
    label = spilt_file(os.path.join(DATAB_ALL_DIR, 'label_list.txt'))
    label_head = ['cid', 'name']

    label = pd.DataFrame(label, columns=label_head)
    label = label.set_index('cid')

    data_test = open(os.path.join(DATAB_ALL_DIR, 'image.txt'))
    data_test = data_test.readlines()

    filenames = []
    for line in data_test:
        filenames.append(line.split()[0])
    fine_label_names = label.iloc[:, 0].tolist()
    print(fine_label_names)
    path = TEST_DATA_DIR
    i = 0
    model = load_model(BAET_CLASSIFY_CKPT_FILE)
    model = Model(inputs=model.input, outputs=model.get_layer(last_layer).output)
    fine_label_names = label.iloc[:, 0].tolist()
    features = []
    for fname in tqdm.tqdm(filenames, desc='Creating Corresponding Data'):
        img = image.load_img(os.path.join(path, fname))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features.append(model.predict(x).reshape(504))
    features = np.asarray(features, dtype='float32')
    features_f = open(TEST_FEATURE_PATH, 'wb')
    pickle.dump({'features': features}, features_f, protocol=4)
    features_f.close()
    print("Done.")


if __name__ == '__main__':
    create_test_visual_feature_main()
