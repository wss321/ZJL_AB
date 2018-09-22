# Code to train the DeViSE Model
from keras.applications import VGG19, VGG16
from keras.applications.vgg19 import preprocess_input

import numpy as np
import tqdm
import os
import pandas as pd
from keras.preprocessing import image
import pickle
from config import SAVE_DIR, NUM_CHANNELS, DATAB_ALL_DIR, DATAB_TRAIN_DIR, DATAA_TRAIN_DIR, DATAA_ALL_DIR
from keras import Model
from keras.models import load_model
from keras_train import BAET_CLASSIFY_CKPT_FILE

if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

np.random.seed(0)


def image_matrix_to_image_array(image_matrix):
    """Gets a image matrix for use in CNNs and converts it back to a CIFAR-100 like matrix"""
    image_size = np.shape(image_matrix)[1]

    image_array = image_matrix.transpose(2, 0, 1).reshape(image_size * image_size * NUM_CHANNELS, )
    return image_array


def image_array_to_image_matrix(image_array):
    """Gets a image matrix from CIFAR-100 and turns it into a matrix suitable for CNN Networks"""
    image_size = int(np.sqrt(np.prod(np.shape(image_array)) / NUM_CHANNELS))

    image_matrix = image_array.reshape(NUM_CHANNELS, image_size, image_size).transpose(1, 2, 0)
    return image_matrix


def spilt_file(file_path):
    return_data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            return_data.append(line.split())
    return return_data


TRAIN_FEATURE_PATH = SAVE_DIR + '/train_AB_visual_features_and_fine_names_kaggle_dense.pkl'


def create_train_visual_feature_main():
    from config import KERAS_MODEL as model
    if model == 'densenet':
        VISUAL_SIZE = 504
        last_layer = 'global_average_pooling2d_1'
    else:
        last_layer = 'batch_normalization_19'
        VISUAL_SIZE = 1024
    label = spilt_file(os.path.join(DATAB_ALL_DIR, 'label_list.txt'))
    label_head = ['cid', 'name']

    label = pd.DataFrame(label, columns=label_head)
    label = label.set_index('cid')

    B_fname_cid = spilt_file(os.path.join(DATAB_ALL_DIR, 'train.txt'))
    A_fname_cid = spilt_file(os.path.join(DATAA_ALL_DIR, 'train.txt'))
    ALL_fname_cid = A_fname_cid + B_fname_cid
    head = ['fname', 'cid']
    ALL_fname_cid = pd.DataFrame(ALL_fname_cid, columns=head)
    ALL_fname_cid = ALL_fname_cid.set_index('fname')
    filenames = ALL_fname_cid.index.tolist()
    i = 0
    # model = VGG19(include_top=True, weights='imagenet')
    # global_average_pooling2d_1
    model = load_model(BAET_CLASSIFY_CKPT_FILE)
    model = Model(inputs=model.input, outputs=model.get_layer(last_layer).output)
    fine_label_names = label.iloc[:, 0].tolist()
    fine_labels = []
    features = []
    for i, fname in enumerate(tqdm.tqdm(filenames, desc='Creating Corresponding Data')):
        if i < len(A_fname_cid):
            path = DATAA_TRAIN_DIR
        else:
            path = DATAB_TRAIN_DIR
        img = image.load_img(os.path.join(path, fname))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features.append(model.predict(x).reshape(VISUAL_SIZE))
        cid = ALL_fname_cid.loc[fname].cid
        fine_name = label.loc[cid]['name']
        fine_labels.append(fine_name)
    features = np.asarray(features, dtype='float32')
    features_f = open(TRAIN_FEATURE_PATH, 'wb')
    pickle.dump({'features': features, 'fine_label_names': fine_labels}, features_f, protocol=4)
    features_f.close()
    print("Done.")


if __name__ == '__main__':
    create_train_visual_feature_main()
