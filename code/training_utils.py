import tensorflow as tf
import pickle
import numpy as np
import time
from tensorflow.contrib.image import rotate

from attr_interface import find_attr_vec
from word2vec_interface import find_word_vec
from config import ALL_LABLES_FILE, WV_OR_ATTR, embedding_size
from create_pickle_file import AB_META
np.random.seed(0)
tf.set_random_seed(0)
meta = pickle.load(open(AB_META, 'rb'))
all_labels = meta['fine_label_names']


def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  # 亮度
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # 饱和度
        image = tf.image.random_hue(image, max_delta=0.2)  # 色相
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # 对比度
    if color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    if color_ordering == 2:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    if color_ordering == 3:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    return tf.clip_by_value(image, 0.0, 1.0)


def random_rotate_image(image):
    """Do random rotate to a image (shape = (height,weigh,channels))"""
    with tf.device('/cpu:0'):
        angle = tf.random_uniform(shape=(1,), minval=-25, maxval=25)
        return rotate(image, angle)


def distort_image(image, image_size, resize):
    """Does random distortion at the training images to avoid overfitting"""
    image = tf.image.resize_images(image, (resize, resize))
    image = tf.random_crop(image, [image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    image = random_rotate_image(image)
    image = tf.image.random_brightness(image,
                                       max_delta=30)
    image = tf.image.random_contrast(image,
                                     lower=0.2, upper=1.8)
    image = distort_color(image, np.random.randint(4))
    # 随机边框裁剪
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    # 随机噪声
    image = tf.slice(image, bbox_begin, bbox_size)
    rand = np.random.randint(100)
    if rand < 20:
        noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=1.0, dtype=tf.float32)
        image = tf.add(image, noise)
    # float_image = tf.image.per_image_standardization(image)
    return image


def distorted_batch(batch, image_size, resize):
    """Creates a distorted image batch"""
    return tf.map_fn(lambda frame: distort_image(frame, image_size, resize), batch)


def print_in_file(string, output_filename=None, verbose=True):
    """Prints a string into a file"""
    if output_filename is not None:
        output_file = open(output_filename, 'a')
        output_file.write(string + '\n')
        output_file.close()
    if verbose is True:
        time.sleep(0.01)
        print(string)
        time.sleep(0.01)


def build_all_labels_repr():
    """Creates a matrix with all labels word2vec representations"""
    all_repr = []
    for label in all_labels:
        if WV_OR_ATTR == 'WV':
            vec = find_word_vec(label)
        else:
            vec = find_attr_vec(label)
        all_repr.append(vec)
    return tf.constant(np.array(all_repr), shape=[len(all_labels), embedding_size], dtype=tf.float32)


def build_all_labels_repr1():
    """Creates a matrix with all labels word2vec representations"""
    all_repr = []
    for label in all_labels:
        if WV_OR_ATTR == 'WV':
            vec = find_word_vec(label)
        else:
            vec = find_attr_vec(label)
        all_repr.append(vec)
    return np.array(all_repr).reshape(len(all_labels), embedding_size)


class Switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:  # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False
