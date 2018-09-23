# Code to predict test data using DEM
import os
import tensorflow as tf
import tqdm
import numpy as np
import pickle
from config import MAX_TO_KEEP, DATAB_ALL_DIR, DEM_MODEL, ZERO_SHOT_CLASSES, SUBMIT_PATH
import pandas as pd
from create_pickle_file import spilt_file, META
from create_test_visual_feature import TEST_FEATURE_PATH
from train_DEM import dem_checkpoint_path, KERAS_MODEL, dem_attr_word2vec_concentrate, TRAIN_FEATURE_PATH, \
    find_concentrate_vec, weight_variable, bias_variable, \
    find_word_vec, find_attr_vec, kNNClassify, dem_hidden_layer
from create_pickle_file import read_pickle_file

np.random.seed(0)


def dem_predict_main():
    def predict1(test_word, test_visual, test_id):
        word_pre = sess.run(left_w1, feed_dict={word_features: test_word})
        test_id = np.squeeze(np.asarray(test_id))
        outpre = [0] * test_visual.shape[0]
        for i in range(test_visual.shape[0]):
            outputLabel = kNNClassify(test_visual[i, :], word_pre, test_id, 1)
            outpre[i] = outputLabel
        return outpre

    def predict2(test_word, test_visual, test_id):
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
    # ------------------------------------------------------
    label = spilt_file(os.path.join(DATAB_ALL_DIR, 'label_list.txt'))
    label_head = ['cid', 'name']

    label = pd.DataFrame(label, columns=label_head)
    label = label.set_index('cid')

    all_class = label.iloc[:, 0].tolist()

    class2cid_dict = read_pickle_file(META)['class2cid']

    data = pickle.load(open(TRAIN_FEATURE_PATH, 'rb'))
    # 训练集包含的类
    fine_names = data['fine_label_names']
    all_train_class_names = list(set(fine_names))
    data = None
    # print(len(set(all_train_class_names)))
    # print(len(all_class))
    # print(all_class)
    # print(all_train_class_names)
    if ZERO_SHOT_CLASSES:
        test_class = list(set(all_class) - set(all_train_class_names))
    else:
        test_class = list(set(all_class))
    print('There are {} test classes:\n{}'.format(len(test_class), test_class))

    if KERAS_MODEL == 'densenet':
        visual_features_size = 504
    else:
        visual_features_size = 1024
    if dem_attr_word2vec_concentrate == 'attr':
        find_vec = find_attr_vec
        embedding_size = 24
    elif dem_attr_word2vec_concentrate == 'word2vec':
        find_vec = find_word_vec
        embedding_size = 300
    else:
        find_vec = find_concentrate_vec
        embedding_size = 324

    # 加载数据
    print("Loading data.")
    data = pickle.load(open(TEST_FEATURE_PATH, 'rb'))
    vf_data = data['features']

    data = None
    print('Done.')
    test_id = range(0, len(test_class))
    word_pro = []

    for name in test_class:
        word_pro.append(find_vec(name))

    test_label = []
    for i in test_class:
        test_label.append(test_class.index(i))

    word_pro = np.asarray(word_pro)

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

    if DEM_MODEL == 1:
        W_left_w1 = weight_variable([embedding_size, visual_features_size])
        b_left_w1 = bias_variable([visual_features_size])
        left_w1 = tf.matmul(word_features, W_left_w1) + b_left_w1
        evaluate_fn = predict1
    else:
        W_left_w1 = weight_variable([embedding_size, dem_hidden_layer])
        W_left_w2 = weight_variable([dem_hidden_layer, visual_features_size])
        b_left_w1 = bias_variable([dem_hidden_layer])
        b_left_w2 = bias_variable([visual_features_size])

        left_w1 = tf.nn.relu(tf.matmul(word_features, W_left_w1) + b_left_w1)
        left_w2 = tf.matmul(left_w1, W_left_w2) + b_left_w2
        evaluate_fn = predict2

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP)

    with tf.Session(config=tfconfig) as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, MODEL_FILE)
        pred = evaluate_fn(word_pro, x_test, test_id)
        class_id = id_num2cid(pred)
        # print(class_id)
    test_fnames_list = spilt_file(os.path.join(DATAB_ALL_DIR, 'image.txt'))
    with open(SUBMIT_PATH, 'w') as submit_f:
        for fname, cid in zip(test_fnames_list, class_id):
            fname = fname[0]
            print(fname, cid)
            string = fname + '\t' + cid + '\n'
            submit_f.write(string)
    print('Prediction Done.\nSubmit file has saved to {}'.format(SUBMIT_PATH))


if __name__ == '__main__':
    dem_predict_main()
