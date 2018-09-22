# Code to predict the DEM
import os
import tensorflow as tf
import tqdm
import numpy as np
from kNN_cosine_or_euclideanl import kNNClassify
from word2vec_interface import find_word_vec
from attr_interface import find_attr_vec
import pickle
from config import SAVE_DIR, MAX_TO_KEEP, META_PKL, SAVE_ALL_LABEL_PKL, ALL_LABLES_FILE, TRAIN_PKL, DATAB_ALL_DIR
import pandas as pd
from create_pickle_file import spilt_file
from train_DEM import dem_checkpoint_path
from create_test_visual_feature import TEST_FEATURE_PATH

np.random.seed(0)
def read_pickle_file(filename):
    """Reads a pickle file using the latin1 encoding"""
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        # u.encoding = 'latin1'
        p = u.load()
    return p


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def cosin_dis(x1, x2):
    x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=1))
    x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=1))
    # 内积
    x3_x4 = tf.reduce_sum(tf.multiply(x1, x2), axis=1)
    dis = tf.divide(x3_x4, tf.multiply(x1_norm, x2_norm))
    return dis


def dem_predict_main():
    def predict(test_word, test_visual, test_id):
        global left_w1
        word_pre = sess.run(left_w1, feed_dict={word_features: test_word})
        test_id = np.squeeze(np.asarray(test_id))
        outpre = [0] * test_visual.shape[0]
        for i in range(test_visual.shape[0]):
            outputLabel = kNNClassify(test_visual[i, :], word_pre, test_id, 1)
            outpre[i] = outputLabel
        return outpre

    def predict2(test_word, test_visual, test_id):
        global left_w2
        word_pre = sess.run(left_w2, feed_dict={word_features: test_word})
        test_id = np.squeeze(np.asarray(test_id))
        outpre = [0] * test_visual.shape[0]
        for i in range(test_visual.shape[0]):
            outputLabel = kNNClassify(test_visual[i, :], word_pre, test_id, 1)
            outpre[i] = outputLabel
        return outpre

    def id_num2cid(id_num):
        cids = []
        for i in id_num:
            class_label = test_class[i]
            cid = class2cid_dict[class_label]
            cids.append(cid)
        return cids

    LATEST_CKPT = tf.train.latest_checkpoint(dem_checkpoint_path)
    MODEL_FILE = LATEST_CKPT
    print('Loaded model at {}'.format(MODEL_FILE))
    # MODEL_FILE = 'DEM_OUTPUT/latest_/epoch1860-val_acc0.0686.ckpt'
    h_dim = 1024
    cosin_lr = 0.0
    eu_lr = 1.0
    reg_r = 1e-4
    # ------------------------------------------------------
    label = spilt_file(os.path.join(DATAB_ALL_DIR, 'label_list.txt'))
    label_head = ['cid', 'name']

    label = pd.DataFrame(label, columns=label_head)
    label = label.set_index('cid')

    all_class = label.iloc[:, 0].tolist()

    fname_cid = spilt_file(os.path.join(DATAB_ALL_DIR, 'train.txt'))
    head = ['fname', 'cid']
    fname_cid = pd.DataFrame(fname_cid, columns=head)
    # 训练集包含的类
    class2cid_dict = read_pickle_file(META_PKL)['class2cid']
    all_train_class = []
    for cid in fname_cid['cid']:
        for key, value in class2cid_dict.items():
            if cid == value:
                cla = key
                if cla not in all_train_class:
                    all_train_class.append(cla)
                break
    # print(len(set(all_train_class)))
    # print(len(all_class))
    # print(all_class)
    # print(all_train_class)
    test_class = list(set(all_class) - set(all_train_class))
    print('There are {} test classes:\n{}'.format(len(test_class), test_class))

    visual_features_size = 504
    attr_or_word2vec = 'word2vec'
    if attr_or_word2vec == 'attr':
        fn = find_attr_vec
        embedding_size = 24
    elif attr_or_word2vec == 'word2vec':
        fn = find_word_vec
        embedding_size = 300

    # 加载数据
    print("Loading data.")
    data = pickle.load(open(TEST_FEATURE_PATH, 'rb'))
    vf_data = data['features']

    data = None
    print('Done.')
    test_id = range(0, len(test_class))
    word_pro = []

    for name in test_class:
        word_pro.append(fn(name))

    test_label = []
    for i in test_class:
        test_label.append(test_class.index(i))

    word_pro = np.asarray(word_pro)

    idx = 0
    x_test = []
    for idx in tqdm.tqdm(range(len(vf_data))):
        x_test.append(vf_data[idx])
    vf_data = None
    data = None

    word_pro = np.asarray(word_pro)
    x_test = np.asarray(x_test)
    test_id = np.asarray(test_id)
    test_label = np.asarray(test_label)

    print('word_pro:{} x_test:{} test_id:{} test_label:{} '.format(
        word_pro.shape,
        x_test.shape,
        test_id.shape,
        test_label.shape))

    tf.reset_default_graph()
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    word_features = tf.placeholder(tf.float32, [None, embedding_size])
    visual_features = tf.placeholder(tf.float32, [None, visual_features_size])

    # W_left_w1 = weight_variable([embedding_size, visual_features_size])
    # b_left_w1 = bias_variable([visual_features_size])
    # left_w1 = tf.nn.relu(tf.matmul(word_features, W_left_w1) + b_left_w1)
    #
    # regular_w = (tf.nn.l2_loss(W_left_w1) + tf.nn.l2_loss(b_left_w1))
    # loss_w = tf.reduce_mean(tf.square(left_w1 - visual_features))

    W_left_w1 = weight_variable([embedding_size, h_dim])
    W_left_w2 = weight_variable([h_dim, visual_features_size])
    b_left_w1 = bias_variable([h_dim])
    b_left_w2 = bias_variable([visual_features_size])

    left_w1 = tf.nn.relu(tf.matmul(word_features, W_left_w1) + b_left_w1)
    left_w2 = tf.nn.relu(tf.matmul(left_w1, W_left_w2) + b_left_w2)

    regular_w = (tf.nn.l2_loss(W_left_w1) + tf.nn.l2_loss(b_left_w1) + tf.nn.l2_loss(W_left_w2) + tf.nn.l2_loss(
        b_left_w2))
    loss_w = tf.add(tf.multiply(eu_lr, tf.reduce_mean(tf.square(left_w2 - visual_features))),
                    tf.multiply(cosin_lr, tf.reduce_mean(tf.square(cosin_dis(left_w2, visual_features)))))

    # 岭回归
    reg_r = 1e-3  # 1e-3
    loss_w += reg_r * regular_w

    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_w)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP)

    with tf.Session(config=tfconfig) as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, MODEL_FILE)
        pred = predict2(word_pro, x_test, test_id)
        class_id = id_num2cid(pred)
        # print(class_id)
    test_fnames_list = spilt_file(os.path.join(DATAB_ALL_DIR, 'image.txt'))
    with open('../data/Submit.txt', 'w') as submit_f:
        for fname, cid in zip(test_fnames_list, class_id):
            fname = fname[0]
            print(fname, cid)
            string = fname + '\t' + cid + '\n'
            submit_f.write(string)
    print('Prediction Done.')


if __name__ == '__main__':
    dem_predict_main()
