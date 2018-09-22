from img_util import image_array_to_image_matrix, resize_image_matrix
import numpy as np
import pickle
import math
import random
from word2vec_interface import find_word_vec
from attr_interface import find_attr_vec
from read_zjl import SAVE_TTRD_PKL, SAVE_NTTRD_PKL, SAVE_VEC_PKL
from config import WV_OR_ATTR, DISTORT
from training_utils import distorted_batch
random.seed(0)
np.random.seed(0)

print('LOADING DATA')
target_train_data = pickle.load(open(SAVE_TTRD_PKL, 'rb'))
not_target_train_data = pickle.load(open(SAVE_NTTRD_PKL, 'rb'))

vectorizer = pickle.load(open(SAVE_VEC_PKL, 'rb'))
print('DATA LOADED')


def adjust_data(image_array, image_size):
    """Resize the image to the needs of the model"""
    image_matrix = image_array_to_image_matrix(image_array)
    resized_image = resize_image_matrix(image_matrix, image_size, image_size)
    return resized_image


def word2vec_batch(word_batch):
    """Takes a word batch and convert it to a dense word vector representation batch"""
    new_batch = []
    for word in word_batch:
        wv = find_word_vec(word)
        new_batch.append(wv)
    return new_batch


def attr_batch(word_batch):
    """Takes a word batch and convert it to a dense attribute vector representation batch"""
    new_batch = []
    for word in word_batch:
        wv = find_attr_vec(word)
        new_batch.append(wv)
    return new_batch


def get_batches(data, size_batch, image_size, use_ebedding=False, send_raw_str=False):
    """Takes a batch of pairs (image, word2vec word) and creates data generators from it
       Arg:
        data format:n * [image, fine_label, coarse_label]
    """
    len_data = len(data)
    num_batches = int(math.floor(len_data / size_batch))

    for i in range(num_batches):
        new_batch = data[i * size_batch:min(len_data, (i + 1) * size_batch)]
        Xs = [adjust_data(b[0], image_size) for b in new_batch]

        raw_Ys = [b[1] for b in new_batch]
        if not use_ebedding:
            Ys = vectorizer.transform(raw_Ys)
        else:
            if WV_OR_ATTR == 'WV':
                Ys = word2vec_batch(raw_Ys)
            else:
                Ys = attr_batch(raw_Ys)

        if not send_raw_str:
            yield [Xs, Ys]
        else:
            yield [Xs, Ys, raw_Ys]


def get_fitdata(data, image_size, use_ebedding=False, send_raw_str=False, distort=True):
    """get data to fit model
           Arg:
            data format:n * [image, fine_label, coarse_label]
        """
    Xs = [adjust_data(b[0], image_size) for b in data]
    if distort:
        Xs = distorted_batch(Xs, image_size)
    raw_Ys = [b[1] for b in data]
    if not use_ebedding:
        Ys = vectorizer.transform(raw_Ys)
    else:
        if WV_OR_ATTR == 'WV':
            Ys = word2vec_batch(raw_Ys)
        else:
            Ys = attr_batch(raw_Ys)

    if not send_raw_str:
        return np.array(Xs), np.array(Ys)
    else:
        return np.array(Xs), np.array(Ys), raw_Ys


def get_fitbatches(session, distort_op, x, data, size_batch, image_size, use=False, send_raw_str=False):
    """Takes a fit batch of pairs (image, word2vec word) and creates data generators from it
       Arg:
        data format:n * [image, fine_label, coarse_label]
    """
    len_data = len(data)
    num_batches = int(math.floor(len_data / size_batch))

    for i in range(num_batches):
        new_batch = data[i * size_batch:min(len_data, (i + 1) * size_batch)]
        Xs = [adjust_data(b[0], image_size) for b in new_batch]
        if DISTORT:
            Xs = session.run(distort_op, feed_dict={x: Xs})
            # print(Xs)
        raw_Ys = [b[1] for b in new_batch]
        if not use:
            Ys = vectorizer.transform(raw_Ys)
        else:
            if WV_OR_ATTR == 'WV':
                Ys = word2vec_batch(raw_Ys)
            else:
                Ys = attr_batch(raw_Ys)

        if not send_raw_str:
            yield np.asarray(Xs).reshape(-1, image_size, image_size, 3), np.asarray(Ys)
        else:
            yield np.asarray(Xs).reshape(-1, image_size, image_size, 3), np.asarray(Ys), raw_Ys
