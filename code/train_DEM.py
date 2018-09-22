# Code to train the DeViSE Model
import os
import random
import tensorflow as tf
import tqdm
import numpy as np
from kNN_cosine_or_euclideanl import kNNClassify
from training_utils import print_in_file
from word2vec_interface import find_word_vec, find_pca_word_vec
from attr_interface import find_attr_vec
import pickle
from config import OUTPUT_FILES_FOLDER, SAVE_DIR, MAX_TO_KEEP, DATAB_ALL_DIR, KERAS_MODEL, DEM_MODEL
from create_train_visual_feature import TRAIN_FEATURE_PATH

tf.set_random_seed(0)
random.seed(0)
np.random.seed(0)

OUTPUT_FILE_NAME = OUTPUT_FILES_FOLDER + '/pretrain_output.txt'
dem_checkpoint_path = 'DEM_OUTPUT/'


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


def cosine_dis(x1, x2):
    x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=1))
    x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=1))
    # 内积
    x3_x4 = tf.reduce_sum(tf.multiply(x1, x2), axis=1)
    dis = tf.divide(x3_x4, tf.multiply(x1_norm, x2_norm))
    return dis


def spilt_class_file(file_path):
    return_data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            return_data.append(line.split()[1])
    return return_data


# text_dict = read_pickle_file('G:\Pycharmworkspace\ZSL_GAN_CVPR18\word_feature_b.pkl')
# key_words = text_dict['words']
# text_vec = text_dict['tf-idf']
# text_dict = None
classes = spilt_class_file(os.path.join(DATAB_ALL_DIR, 'label_list.txt'))

# print(key_words)


# def find_text_vec(class_name):
#     idx = classes.index(class_name)
#     return text_vec[idx]

attr_or_word2vec = 'word2vec'  # 'attr' 'word2vec'


def train_dem_main(epoches=100000):
    def data_iterator(batch_size):
        """ A simple data iterator """
        batch_idx = 0
        # shuffle labels and features
        idxs = np.arange(0, len(train_vs))
        np.random.shuffle(idxs)
        shuf_visual = train_vs[idxs]
        shuf_word = train_wordvec[idxs]
        if batch_size > len(train_vs):
            batch_size = len(train_vs)
        for batch_idx in range(0, len(train_vs), batch_size):
            visual_batch = shuf_visual[batch_idx:batch_idx + batch_size]
            visual_batch = visual_batch.astype("float32")
            word_batch = shuf_word[batch_idx:batch_idx + batch_size]
            yield word_batch, visual_batch

    if KERAS_MODEL == 'densenet':
        visual_features_size = 504
    else:
        visual_features_size = 1024
    # -----training parameter-----
    LOAD_CKPT = False
    batch_size = 30000
    split_rate = 0.8
    cosin_lr = 0.0
    eu_lr = 1.0
    reg_r = 1e-5

    # ---------------------------

    if attr_or_word2vec == 'attr':
        fn = find_attr_vec
        embedding_size = 24
    elif attr_or_word2vec == 'word2vec':
        fn = find_word_vec
        embedding_size = 300
    # elif attr_or_word2vec == 'text':
    #     fn = find_text_vec
    #     embedding_size = text_vec.shape[1]

    # 加载数据
    print("Loading data.")
    data = pickle.load(open(TRAIN_FEATURE_PATH, 'rb'))
    vf_data = data['features']
    fine_names = data['fine_label_names']
    all_names = list(set(fine_names))
    # random.shuffle(all_names)
    data = None
    print('Done.')
    # test_id = random.sample(range(0, len(all_names)), int(len(all_names) * split_rate))
    test_id = list(i for i, _ in enumerate(all_names[int(len(all_names) * split_rate):]))
    word_pro = []

    test_names = []
    for i in test_id:
        name = all_names[i]
        test_names.append(name)
        word_pro.append(fn(name))

    test_label = []
    for i in fine_names:
        if i in test_names:
            test_label.append(all_names.index(i))

    word_pro = np.asarray(word_pro)
    print(test_names)
    print(len(test_names))

    # 分离训练集和测试集
    idx = 0
    x_test = []
    train_data = []
    for fine_name in tqdm.tqdm(fine_names, ):
        if fine_name not in test_names:
            train_data.append([np.asarray(vf_data[idx]), fn(fine_name)])
            # print(type(vf_data[idx]))
        else:
            x_test.append(vf_data[idx])
        idx += 1
    vf_data = None
    data = None

    train_wordvec = []
    train_vs = []

    random.shuffle(train_data)
    # train_data = np.asarray(train_data)

    for i in train_data:
        train_wordvec.append(i[1])
        train_vs.append(i[0])
    train_wordvec = np.asarray(train_wordvec).reshape(len(train_data), embedding_size)
    train_vs = np.asarray(train_vs).reshape(len(train_data), visual_features_size)
    train_data = None
    word_pro = np.asarray(word_pro)
    x_test = np.asarray(x_test)
    test_id = np.asarray(test_id)
    test_label = np.asarray(test_label)

    def compute_accuracy1(test_word, test_visual, test_id, test_label):
        # global left_w1
        word_pre = sess.run(left_w1, feed_dict={word_features: test_word})
        test_id = np.squeeze(np.asarray(test_id))
        outpre = [0] * test_visual.shape[0]
        test_label = np.squeeze(np.asarray(test_label))
        test_label = test_label.astype("float32")
        for i in range(test_visual.shape[0]):
            outputLabel = kNNClassify(test_visual[i, :], word_pre, test_id, 1)
            outpre[i] = outputLabel
        print(outpre)
        correct_prediction = tf.equal(outpre, test_label)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={
            word_features: test_word, visual_features: test_visual})
        return result

    def compute_accuracy2(test_word, test_visual, test_id, test_label):
        # global left_w2
        word_pre = sess.run(left_w2, feed_dict={word_features: test_word})
        test_id = np.squeeze(np.asarray(test_id))
        outpre = [0] * test_visual.shape[0]
        test_label = np.squeeze(np.asarray(test_label))
        test_label = test_label.astype("float32")
        for i in range(test_visual.shape[0]):
            outputLabel = kNNClassify(test_visual[i, :], word_pre, test_id, 1)
            outpre[i] = outputLabel
        print(outpre)
        correct_prediction = tf.equal(outpre, test_label)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={
            word_features: test_word, visual_features: test_visual})
        return result

    print(
        'train_wordvec:{} train_vs:{} word_pro:{} x_test:{} test_id:{} test_label:{} '.format(train_wordvec.shape,
                                                                                              train_vs.shape,
                                                                                              word_pro.shape,
                                                                                              x_test.shape,
                                                                                              test_id.shape,
                                                                                              test_label.shape))

    tf.reset_default_graph()
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True

    if not os.path.exists(dem_checkpoint_path):
        os.makedirs(dem_checkpoint_path)
    LOAD_CKPT_FILE = tf.train.latest_checkpoint(dem_checkpoint_path)

    word_features = tf.placeholder(tf.float32, [None, embedding_size])
    visual_features = tf.placeholder(tf.float32, [None, visual_features_size])
    if DEM_MODEL == 1:
        W_left_w1 = weight_variable([train_wordvec.shape[1], visual_features_size])
        b_left_w1 = bias_variable([visual_features_size])
        left_w1 = tf.matmul(word_features, W_left_w1) + b_left_w1

        regular_w = (tf.nn.l2_loss(W_left_w1) + tf.nn.l2_loss(b_left_w1))
        loss_w = tf.add(tf.multiply(eu_lr, tf.reduce_mean(tf.square(left_w1 - visual_features))),
                        tf.multiply(cosin_lr, tf.reduce_mean(tf.square(cosine_dis(left_w1, visual_features)))))
        evaluate_fn = compute_accuracy1
    else:
        W_left_w1 = weight_variable([train_wordvec.shape[1], 1024])
        W_left_w2 = weight_variable([1024, visual_features_size])
        b_left_w1 = bias_variable([1024])
        b_left_w2 = bias_variable([visual_features_size])

        left_w1 = tf.nn.relu(tf.matmul(word_features, W_left_w1) + b_left_w1)
        left_w2 = tf.matmul(left_w1, W_left_w2) + b_left_w2

        regular_w = (tf.nn.l2_loss(W_left_w1) + tf.nn.l2_loss(b_left_w1) + tf.nn.l2_loss(W_left_w2) + tf.nn.l2_loss(
            b_left_w2))
        loss_w = tf.add(tf.multiply(eu_lr, tf.reduce_mean(tf.square(left_w2 - visual_features))),
                        tf.multiply(cosin_lr, tf.reduce_mean(tf.square(cosine_dis(left_w2, visual_features)))))
        evaluate_fn = compute_accuracy2

    # 岭回归
    loss_w += reg_r * regular_w

    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_w)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP)

    # Start Tensorflow session
    with tf.Session(config=tfconfig) as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        print_in_file('--------------------------Start Training--------------------------', OUTPUT_FILE_NAME)
        # Load the pretrained weights into the non-trainable layer
        if LOAD_CKPT and LOAD_CKPT_FILE:
            print_in_file('Loading checkpoint at {}'.format(LOAD_CKPT_FILE), OUTPUT_FILE_NAME)
            saver.restore(sess, 'DEM_OUTPUT/latest_/epoch1860-val_acc0.0686.ckpt')  # LOAD_CKPT_FILE
        step = 0
        for epoch in range(epoches):
            iter_ = data_iterator(batch_size)
            for word_batch_val, visual_batch_val in iter_:
                sess.run(train_step, feed_dict={word_features: word_batch_val, visual_features: visual_batch_val})
                train_loss = sess.run(loss_w,
                                      feed_dict={word_features: word_batch_val, visual_features: visual_batch_val})
                print_in_file("\nEpoch:{}/{} - Train loss = {}".format(epoch, epoches, train_loss), OUTPUT_FILE_NAME)
                acc = evaluate_fn(word_pro, x_test, test_id, test_label)
                print_in_file(
                    "Epoch:{}/{} - Val acc = {}".format(epoch, epoches, acc), OUTPUT_FILE_NAME)
                if epoch % 10 == 0:
                    saver.save(sess=sess,
                               save_path='{}/epoch{}-val_acc{:.4f}.ckpt'.format(dem_checkpoint_path, epoch, float(acc)))


if __name__ == '__main__':
    train_dem_main()
