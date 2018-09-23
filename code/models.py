import tensorflow as tf
import numpy as np
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.slim.nets import resnet_v2
from config import nb_block, training_flag, depth, dropout_rate

np.random.seed(0)
tf.set_random_seed(0)


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1, verbose_shapes=False, batch_norm=False):
    """Convolution function that can be split in multiple GPUs"""
    # Get number of input chennels
    input_channels = int(x.get_shape()[-1])

    if verbose_shapes:
        print('INPUT_CHANNELS', input_channels)
        print('X SHAPE conv', x.get_shape())

    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        try:
            weights = tf.get_variable('weights',
                                      shape=[filter_height, filter_width, input_channels / groups, num_filters],
                                      trainable=True,
                                      initializer=tf.contrib.layers.xavier_initializer())
        except:
            tf.get_variable_scope().reuse_variables()
            weights = tf.get_variable('weights',
                                      shape=[filter_height, filter_width, input_channels / groups, num_filters],
                                      trainable=True,
                                      initializer=tf.contrib.layers.xavier_initializer())

        try:
            biases = tf.get_variable('biases', shape=[num_filters], trainable=True,
                                     initializer=tf.contrib.layers.xavier_initializer())
        except:
            tf.get_variable_scope().reuse_variables()
            biases = tf.get_variable('biases', shape=[num_filters], trainable=True,
                                     initializer=tf.contrib.layers.xavier_initializer())

        if groups == 1:
            conv = convolve(x, weights)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            conv = tf.concat(axis=3, values=output_groups)

        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        if batch_norm:
            norm = lrn(bias, 2, 2e-05, 0.75, name=scope.name)
            relu = tf.nn.relu(norm, name=scope.name)
        else:
            relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(x, num_in, num_out, name, relu=True, use_biases=True):
    """Full connected layer"""
    with tf.variable_scope(name) as scope:
        try:
            weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True,
                                      initializer=tf.contrib.layers.xavier_initializer())
        except:
            tf.get_variable_scope().reuse_variables()
            weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True,
                                      initializer=tf.contrib.layers.xavier_initializer())

        if use_biases:
            try:
                biases = tf.get_variable('biases', [num_out], trainable=True,
                                         initializer=tf.contrib.layers.xavier_initializer())
            except:
                tf.get_variable_scope().reuse_variables()
                biases = tf.get_variable('biases', [num_out], trainable=True,
                                         initializer=tf.contrib.layers.xavier_initializer())

            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
        else:
            act = tf.matmul(x, weights)

        if relu:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x,
             name, padding='SAME', verbose_shapes=False):
    """Max pool layer"""
    if verbose_shapes:
        print('X SHAPE maxpool', x.get_shape())

    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0, verbose_shapes=False):
    """Batch normalization"""
    if verbose_shapes:
        print('X SHAPE lrn', x.get_shape())

    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def avg_pool(x, filter_height, filter_width, stride_y, stride_x,
             name, padding='SAME', verbose_shapes=False):
    """Average pooling layer"""
    if verbose_shapes:
        print('X SHAPE avgpool', x.get_shape())

    return tf.nn.avg_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def normalize_images(x):
    """Normalize images before feeding into a CNN"""
    return tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x)


def dropout(x, keep_prob):
    """Dropout layer"""
    return tf.nn.dropout(x, keep_prob)


class AlexNet(object):
    """AlexNet model"""

    def __init__(self, x, num_classes):
        self.X = x
        self.NUM_CLASSES = num_classes
        self.create()

    def create(self):
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        normalized_images = normalize_images(self.X)
        self.conv1_1 = conv(normalized_images, 5, 5, 64, 1, 1, padding='SAME', name='conv1_1')
        self.conv1_2 = conv(self.conv1_1, 5, 5, 64, 1, 1, padding='SAME', name='conv1_2')
        norm1 = lrn(self.conv1_2, 2, 2e-05, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='SAME', name='pool1')

        # 2nd Layer: Conv (w ReLu) -> Lrn -> Poolwith 2 groups
        self.conv2_1 = conv(pool1, 3, 3, 128, 1, 1, groups=2, name='conv2_1')
        self.conv2_2 = conv(pool1, 3, 3, 128, 1, 1, groups=2, name='conv2_2')
        norm2 = lrn(self.conv2_1, 2, 2e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='SAME', name='pool2')
        # 3rd Layer: Conv (w ReLu) -> Lrn -> Poolwith 2 groups
        self.conv3 = conv(pool2, 5, 5, 256, 1, 1, groups=2, name='conv3')
        norm3 = lrn(self.conv3, 2, 2e-05, 0.75, name='norm3')
        pool3 = max_pool(norm3, 3, 3, 2, 2, padding='SAME', name='pool3')

        self.conv4 = conv(pool3, 3, 3, 128, 1, 1, padding='SAME', name='conv4')
        norm4 = lrn(self.conv4, 2, 2e-05, 0.75, name='norm4')
        pool4 = max_pool(norm4, 3, 3, 2, 2, padding='SAME', name='pool4')

        # 3th Layer: Flatten -> FC (w ReLu) -> Dropout
        self.flattened = flatten(pool4)  # tf.reshape(pool2, [-1, 4 * 4 * 64])
        self.fc3 = fc(self.flattened, self.flattened.get_shape().as_list()[1], 384, name='fc3')

        # 4th Layer: FC (w ReLu) -> Dropout
        self.fc4 = fc(self.fc3, self.fc3.get_shape().as_list()[1], 192, name='fc4')

        # 5th Layer: FC and return unscaled activations
        # (for tf.nn.softmax_cross_entropy_with_logits)
        self.fc5 = fc(self.fc4, self.fc4.get_shape().as_list()[1], self.NUM_CLASSES, relu=False, name='fc5')


class Composite_model(object):
    """Visual-semantic embedding"""

    def __init__(self, x, num_classes, word2vec_size, model_choose='vgg19', dropout_rate=0.5):
        self.X = x
        self.NUM_CLASSES = num_classes
        self.WORD2VEC_SIZE = word2vec_size
        self.model_choose = model_choose

        if self.model_choose == 'vgg19':
            self.image_repr_model = VGG19(self.X, 0.5, self.NUM_CLASSES)
        elif self.model_choose == 'alexnet':
            self.image_repr_model = AlexNet(self.X, self.NUM_CLASSES)
        elif self.model_choose == 'densenet':
            self.image_repr_model = DenseNet(self.X, self.NUM_CLASSES, nb_blocks=nb_block, filters=depth,
                                             dropout_rate=dropout_rate,
                                             training=tf.constant(training_flag, dtype=tf.bool))
        elif self.model_choose == 'resnet_v2_50':
            self.image_repr_model = ResNet_v2_50(self.X, self.NUM_CLASSES,
                                                 training=tf.constant(training_flag, dtype=tf.bool))
        self.create()

    def create(self):
        if self.model_choose == 'vgg19':
            self.image_repr = self.image_repr_model.fc7
            self.projection_layer = fc(self.image_repr, self.image_repr.get_shape().as_list()[1], self.WORD2VEC_SIZE,
                                       name='proj', relu=False,
                                       use_biases=True)
        elif self.model_choose == 'alexnet':
            self.image_repr = self.image_repr_model.fc4
            self.projection_layer = fc(self.image_repr, self.image_repr.get_shape().as_list()[1], self.WORD2VEC_SIZE,
                                       name='proj', relu=False,
                                       use_biases=True)
        elif self.model_choose == 'densenet':
            self.image_repr = self.image_repr_model.flatten
            self.projection_layer = fc(self.image_repr, self.image_repr.get_shape().as_list()[1], self.WORD2VEC_SIZE,
                                       name='proj', relu=False,
                                       use_biases=True)
        elif self.model_choose == 'resnet_v2_50':
            self.image_repr = self.image_repr_model.flat
            self.projection_layer = fc(self.image_repr, self.image_repr.get_shape().as_list()[1], self.WORD2VEC_SIZE,
                                       name='proj', relu=False,
                                       use_biases=True)


class Reverse_model(object):
    def __init__(self, x, word2vec_size, image_size):
        self.X = x
        self.WORD2VEC_SIZE = word2vec_size
        self.IMAGE_SIZE = image_size
        self.create()

    def create(self):
        self.L1 = fc(self.X, self.WORD2VEC_SIZE, (self.IMAGE_SIZE * self.IMAGE_SIZE * 3) / 2, name='L1', relu=True)
        self.L2 = fc(self.L1, (self.IMAGE_SIZE * self.IMAGE_SIZE * 3) / 2, self.IMAGE_SIZE * self.IMAGE_SIZE * 3,
                     name='L2', relu=False)
        self.final_image = tf.reshape(self.L2, (-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 3))


class VGG19(object):
    """VGG19 model"""

    def __init__(self, x, keep_prob, num_classes):
        self.X = x
        self.KEEP_PROB = keep_prob
        self.NUM_CLASSES = num_classes
        self.create()

    def create(self):
        normalized_images = normalize_images(self.X)

        conv1_1 = conv(normalized_images, 3, 3, 64, 1, 1, padding='SAME', name='conv1_1', batch_norm=True)
        norm1_1 = lrn(conv1_1, 2, 2e-05, 0.75, name='norm1_1')
        conv1_2 = conv(norm1_1, 3, 3, 64, 1, 1, padding='SAME', name='conv1_2', batch_norm=True)
        norm1_2 = lrn(conv1_2, 2, 2e-05, 0.75, name='norm1_2')
        pool1 = max_pool(norm1_2, 2, 2, 2, 2, padding='SAME', name='pool1')

        norm2_1 = lrn(pool1, 2, 2e-05, 0.75, name='norm2_1')
        conv2_1 = conv(norm2_1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_1', batch_norm=True)
        norm2_2 = lrn(conv2_1, 2, 2e-05, 0.75, name='norm2_2')
        conv2_2 = conv(norm2_2, 3, 3, 128, 1, 1, padding='SAME', name='conv2_2', batch_norm=True)
        pool2 = max_pool(conv2_2, 2, 2, 2, 2, padding='SAME', name='pool2')

        norm3_1 = lrn(pool2, 2, 2e-05, 0.75, name='norm3_1')
        conv3_1 = conv(norm3_1, 3, 3, 256, 1, 1, padding='SAME', name='conv3_1', batch_norm=True)
        norm3_2 = lrn(conv3_1, 2, 2e-05, 0.75, name='norm3_2')
        conv3_2 = conv(norm3_2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_2', batch_norm=True)
        norm3_3 = lrn(conv3_2, 2, 2e-05, 0.75, name='norm3_3')
        conv3_3 = conv(norm3_3, 3, 3, 256, 1, 1, padding='SAME', name='conv3_3', batch_norm=True)
        norm3_4 = lrn(conv3_3, 2, 2e-05, 0.75, name='norm3_4')
        conv3_4 = conv(norm3_4, 3, 3, 256, 1, 1, padding='SAME', name='conv3_4', batch_norm=True)
        pool3 = max_pool(conv3_4, 2, 2, 2, 2, padding='SAME', name='pool3')

        norm4_1 = lrn(pool3, 2, 2e-05, 0.75, name='norm4_1')
        conv4_1 = conv(norm4_1, 3, 3, 512, 1, 1, padding='SAME', name='conv4_1', batch_norm=True)
        norm4_2 = lrn(conv4_1, 2, 2e-05, 0.75, name='norm4_2')
        conv4_2 = conv(norm4_2, 3, 3, 512, 1, 1, padding='SAME', name='conv4_2', batch_norm=True)
        norm4_3 = lrn(conv4_2, 2, 2e-05, 0.75, name='norm4_3')
        conv4_3 = conv(norm4_3, 3, 3, 512, 1, 1, padding='SAME', name='conv4_3', batch_norm=True)
        norm4_4 = lrn(conv4_3, 2, 2e-05, 0.75, name='norm4_4')
        conv4_4 = conv(norm4_4, 3, 3, 512, 1, 1, padding='SAME', name='conv4_4', batch_norm=True)
        pool4 = max_pool(conv4_4, 2, 2, 2, 2, padding='SAME', name='pool4')

        flattened_shape = np.prod([s.value for s in pool4.get_shape()[1:]])
        flattened = tf.reshape(pool4, [-1, flattened_shape], name='flatenned')
        fc6 = fc(flattened, flattened_shape, 4096, name='fc6')
        self.fc7 = fc(fc6, 4096, 4096, name='fc7')
        self.fc8 = fc(self.fc7, 4096, self.NUM_CLASSES, relu=False, name='fc8')


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
                                   padding='SAME')
        return network


def Global_Average_Pooling(x, stride=1):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def Drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def Relu(x):
    return tf.nn.relu(x)


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Concatenation(layers):
    return tf.concat(layers, axis=3)


def Linear(x, units):
    return tf.layers.dense(inputs=x, units=units, name='linear')


class DenseNet(object):
    def __init__(self, x, num_classes, nb_blocks, filters, dropout_rate, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.NUM_CLASSES = num_classes
        self.X = x
        self.dropout_rate = dropout_rate
        self.create()

    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3, 3], layer_name=scope + '_conv2')
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2, 2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def create(self):
        x = conv_layer(self.X, filter=2 * self.filters, kernel=[7, 7], stride=2, layer_name='conv0')
        # x = Max_Pooling(x, pool_size=[3,3], stride=2)

        """
        for i in range(self.nb_blocks) :
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))
        """

        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        # 100 Layer
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        self.flatten = flatten(x)
        self.linear = Linear(self.flatten, units=self.NUM_CLASSES)


class ResNet_v2_50(object):
    def __init__(self, x, num_classes, training):
        self.training = training
        self.NUM_CLASSES = num_classes
        self.X = x
        self.create()

    def create(self):
        arg_scope = resnet_v2.resnet_arg_scope()
        with resnet_v2.arg_scope(arg_scope):
            self.net, self.end_points = resnet_v2.resnet_v2_50(self.X, self.NUM_CLASSES, is_training=self.training)
            self.pool5 = tf.get_default_graph().get_tensor_by_name('resnet_v2_50/pool5:0')
            # names = [op.name for op in tf.get_default_graph().get_operations()]
            # print(names)
            self.flat = flatten(self.pool5)
            self.fc = fc(self.flat, self.flat.get_shape().as_list()[1], 512, relu=False, name='resnet_fc')
