# Code to train classifier


def keras_train_main():
    import random
    import os
    import tensorflow as tf
    from keras import backend as K
    from training_utils import distorted_batch

    from config import num_classes, batch_size, MODEL_DIR, KERAS_MODEL, \
        OPTIMIZER, initial_learning_rate, num_epochs
    from data_generator import DataGenerator
    import keras
    from keras.models import load_model
    from keras.optimizers import SGD, Adam
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import TensorBoard, EarlyStopping
    from config import TB_LOG, BAET_CLASSIFY_CKPT_FILE, IMAGE_SIZE, NUM_CHANNELS, TRAINING_DIR
    from batch_making import get_fitdata
    from densenet_keras import DenseNet
    from vgg_bn import VGG_BN
    from batch_making import target_train_data
    random.seed(0)
    tf.set_random_seed(0)
    LOAD_CKPT = False
    tf.reset_default_graph()
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    session = tf.Session(config=tfconfig)
    K.set_session(session)
    x = K.placeholder(dtype=tf.float32, shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))

    # 回调函数
    tensorboard = TensorBoard(log_dir=TB_LOG)
    checkpoint = ModelCheckpoint(filepath=BAET_CLASSIFY_CKPT_FILE, monitor='val_acc', mode='auto',
                                 save_best_only='True')
    losscalback = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=1, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    callback_lists = [earlystop, checkpoint, losscalback, tensorboard]

    if OPTIMIZER == 'adam':
        optm = Adam(initial_learning_rate)
    else:
        optm = SGD(lr=initial_learning_rate)

    if LOAD_CKPT and os.path.exists(MODEL_DIR):
        print("LOADING MODEL AT {}".format(MODEL_DIR))
        model = load_model(MODEL_DIR)
        resize = 64
        distort_op = distorted_batch(x, IMAGE_SIZE, resize)
        print("DONE.")
    elif KERAS_MODEL == 'densenet':
        IMAGE_SIZE = 64
        model = DenseNet((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), depth=64, nb_dense_block=4,
                         growth_rate=12, bottleneck=True, dropout_rate=0.1, reduction=0.0, classes=num_classes)
        resize = 64
        distort_op = distorted_batch(x, IMAGE_SIZE, resize)
        model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])  #
        model.summary()
    else:
        IMAGE_SIZE = 64
        model = VGG_BN(num_classes, norm_rate=0.0)
        resize = 64
        distort_op = distorted_batch(x, IMAGE_SIZE, resize)
        model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])  #
        model.summary()

    # 加载数据
    TRAIN_TEST_SPLIT_RATE = 0.8
    # for i in target_train_data:
    #     print(i[1])
    random.shuffle(target_train_data)

    data_len = len(target_train_data)
    train_data = target_train_data[:int(data_len * TRAIN_TEST_SPLIT_RATE)]
    vali_data = target_train_data[int(data_len * TRAIN_TEST_SPLIT_RATE):]
    print(len(train_data))
    print(len(vali_data))

    target_train_data = None  # release space
    not_target_train_data = None
    train_generator = DataGenerator(session, distort_op, x, train_data, batch_size=batch_size, distort=False,
                                    shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_data = None
    x_vali, y_vali = get_fitdata(vali_data, IMAGE_SIZE, use_ebedding=False, send_raw_str=False, distort=False)
    vali_data = None

    h = model.fit_generator(generator=train_generator, verbose=1,
                            epochs=num_epochs, callbacks=callback_lists,
                            validation_data=(x_vali, y_vali))
    # model.evaluate(x_vali, y_vali)
    with open(os.path.join(TRAINING_DIR, 'train_history,txt'), 'a') as f:
        f.write(str(h.history) + '\n')
    model.save(MODEL_DIR)


if __name__ == '__main__':
    keras_train_main()
