import keras
from img_util import image_array_to_image_matrix, resize_image_matrix
import numpy as np
import pickle
import math
import random
from word2vec_interface import find_word_vec
from attr_interface import find_attr_vec
from config import SAVE_TTRD_PKL, SAVE_NTTRD_PKL, SAVE_VEC_PKL, WV_OR_ATTR, DISTORT
from training_utils import distorted_batch

random.seed(0)
np.random.seed(0)

print('LOADING DATA')
target_train_data = pickle.load(open(SAVE_TTRD_PKL, 'rb'))
not_target_train_data = pickle.load(open(SAVE_NTTRD_PKL, 'rb'))

vectorizer = pickle.load(open(SAVE_VEC_PKL, 'rb'))
print('DATA LOADED')
#
# used = ['goldfish', 'tarantula', 'centipede', 'goose', 'koala', 'jellyfish', 'coral', 'snail', 'slug', 'lobster',
#         'salamander', 'stork', 'penguin', 'albatross', 'dugong', 'terrier', 'retriever', 'shepherd', 'bullfrog',
#         'poodle', 'tabby', 'cat', 'cougar', 'lion', 'bear', 'ladybug', 'fly', 'bee', 'frog', 'grasshopper', 'cockroach',
#         'mantis', 'dragonfly', 'monarch', 'butterfly', 'pig', 'hog', 'alligator', 'ox', 'bison', 'bighorn', 'gazelle',
#         'camel', 'orangutan', 'chimpanzee', 'baboon', 'elephant', 'panda', 'constrictor', 'trilobite', 'scorpion',
#         'player', 'teddy', 'car', 'go-kart', 'gondola', 'lifeboat', 'limousine', 'van', 'bus', 'sportscar', 'trunks',
#         'rickshaw', 'wagon', 'train', 'convertible', 'crane', 'tractor', 'trolleybus', 'coat', 'kimono', 'uniform',
#         'miniskirt', 'poncho', 'sock', 'sombrero', 'gown', 'apron', 'bikini', 'bowtie', 'cardigan', 'stocking',
#         'vestment', 'cucumber', 'potato', 'cauliflower', 'pepper', 'mushroom', 'orange', 'banana', 'pomegranate',
#         'acorn', 'bottle', 'teapot', 'jug', 'spoon', 'plate', 'pan', 'mask', 'hourglass', 'ipod', 'lampshade', 'mower',
#         'compass', 'nail', 'brace', 'meter', 'phone', 'plunger', 'pole', 'wheel', 'projectile', 'reel', 'refrigerator',
#         'remote-control', 'snorkel', 'heater', 'web', 'stopwatch', 'stick', 'abacus', 'bannister', 'barrel', 'bathtub',
#         'beaker', 'binoculars', 'broom', 'candle', 'machine', 'keyboard', 'torch', 'turnstile', 'wok', 'drumstick',
#         'dumbbell', 'flagpole', 'fountain', 'maypole', 'obelisk', 'oboe', 'organ', 'fence', 'bag', 'chair', 'ball',
#         'sandal', 'scoreboard', 'sunglasses', 'bridge', 'altar', 'backpack', 'barbershop', 'barn', 'basketball',
#         'beacon', 'birdhouse', 'brass', 'bucket', 'shop', 'cannon', 'chain', 'widow', 'chest', 'dwelling',
#         'confectionery', 'dam', 'desk', 'table', 'syringe', 'thatch', 'arch', 'umbrella', 'viaduct', 'volleyball',
#         'tower', 'book', 'guacamole', 'icecream', 'lolly', 'pretzel', 'lemon', 'loaf', 'pizza', 'potpie', 'espresso',
#         'alp', 'cliff', 'reef', 'lakeside', 'seashore']
# v = vectorizer.transform(used)
# print(v)
# print(v.shape)
# print(vectorizer.inverse_transform(np.asarray([v[0]])))


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


def get_fitdata(data, image_size, use=False, send_raw_str=False, distort=True):
    """get data to fit model
           Arg:
            data format:n * [image, fine_label, coarse_label]
        """
    Xs = [adjust_data(b[0], image_size) for b in data]
    if distort:
        Xs = distorted_batch(Xs, image_size)
    raw_Ys = [b[1] for b in data]
    if not use:
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


def get_fitbatches(data, size_batch, image_size, use=False, send_raw_str=False, distort=True):
    """Takes a fit batch of pairs (image, word2vec word) and creates data generators from it
       Arg:
        data format:n * [image, fine_label, coarse_label]
    """
    len_data = len(data)
    num_batches = int(math.floor(len_data / size_batch))

    for i in range(num_batches):
        new_batch = data[i * size_batch:min(len_data, (i + 1) * size_batch)]
        Xs = [adjust_data(b[0], image_size) for b in new_batch]
        if distort:
            Xs = distorted_batch(Xs, image_size)
        raw_Ys = [b[1] for b in new_batch]
        if not use:
            Ys = vectorizer.transform(raw_Ys)
        else:
            if WV_OR_ATTR == 'WV':
                Ys = word2vec_batch(raw_Ys)
            else:
                Ys = attr_batch(raw_Ys)

        if not send_raw_str:
            yield np.array(Xs), np.array(Ys)
        else:
            yield np.array(Xs), np.array(Ys), raw_Ys


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, session, distort_op, x_pc, datas, batch_size=32, shape=(64, 64, 3),
                 n_classes=190, shuffle=True, distort=True, use_embedding=False):
        """Initialization"""
        self.shape = shape
        self.batch_size = batch_size
        self.datas = datas
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.distort = distort
        self.use_embedding = use_embedding
        self.session = session
        self.distort_op = distort_op
        self.x_pc = x_pc
        self.indexes = np.arange(len(self.datas))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.datas) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        batch_data = [self.datas[k] for k in batch_indexs]

        # Generate data
        X, y = self.__data_generation(batch_data)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_data):
        """Generates data containing batch_size samples"""
        # Generate data
        Xs = [adjust_data(b[0], self.shape[1]) for b in batch_data]
        if self.distort:
            Xs = self.session.run(self.distort_op, feed_dict={self.x_pc: Xs})
        raw_Ys = [b[1] for b in batch_data]
        if not self.use_embedding:
            Ys = vectorizer.transform(raw_Ys)
        else:
            if WV_OR_ATTR == 'WV':
                Ys = word2vec_batch(raw_Ys)
            else:
                Ys = attr_batch(raw_Ys)
        return np.asarray(Xs).reshape(-1, self.shape[0], self.shape[1], self.shape[2]), np.asarray(Ys)
