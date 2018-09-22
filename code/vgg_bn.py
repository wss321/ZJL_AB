from keras import Model
from keras.layers import Flatten, Dense, Input, Convolution2D, MaxPooling2D, BatchNormalization, Activation
from keras import regularizers

# size of pooling area for max pooling
pool_size = (2, 2)
stride = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


def VGG_BN(num_class, norm_rate=0.0):
    inputs = Input(shape=(64, 64, 3,), name='input')
    x = BatchNormalization()(inputs)
    # block 1
    x = Convolution2D(64, kernel_size=(7, 7), padding='same', name='Convb1_1',
                      kernel_regularizer=regularizers.l2(norm_rate))(x)

    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Convolution2D(64, kernel_size=(5, 5), padding='same', name='Convb1_2',
                      kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=pool_size, strides=(2, 2), name='Pool1')(x)
    # block 2
    x = Convolution2D(128, kernel_size=kernel_size, padding='same', name='Convb2_1',
                      kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(128, kernel_size=kernel_size, padding='same', name='Convb2_2',
                      kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=pool_size, strides=stride, name='Pool2')(x)
    # block 3
    x = Convolution2D(256, kernel_size=kernel_size, padding='same', name='Convb3_1',
                      kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256, kernel_size=kernel_size, padding='same', name='Convb3_2',
                      kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Convolution2D(256, kernel_size=kernel_size, padding='same', name='Convb3_3',
                      kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256, kernel_size=kernel_size, padding='same', name='Convb3_4',
                      kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=pool_size, strides=stride, name='Pool3')(x)
    # block 4
    x = Convolution2D(512, kernel_size=kernel_size, padding='same', name='Convb4_1',
                      kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Convolution2D(512, kernel_size=kernel_size, padding='same', name='Convb4_2',
                      kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Convolution2D(512, kernel_size=kernel_size, padding='same', name='Convb4_3',
                      kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(512, kernel_size=kernel_size, padding='same', name='Convb4_4',
                      kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=pool_size, strides=stride, name='Pool4')(x)
    # block 5
    x = Convolution2D(512, kernel_size=kernel_size, padding='same', name='Convb5_1',
                      kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Convolution2D(512, kernel_size=kernel_size, padding='same', name='Convb5_2',
                      kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Convolution2D(512, kernel_size=kernel_size, padding='same', name='Convb5_3',
                      kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(512, kernel_size=kernel_size, padding='same', name='Convb5_4',
                      kernel_regularizer=regularizers.l2(norm_rate))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=pool_size, strides=stride, name='Pool5')(x)
    # # block 6
    # x = Convolution2D(1024, kernel_size=kernel_size, padding='same', name='Convb6_1',
    #                   kernel_regularizer=regularizers.l2(norm_rate))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Convolution2D(1024, kernel_size=kernel_size, padding='same', name='Convb6_2',
    #                   kernel_regularizer=regularizers.l2(norm_rate))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Convolution2D(1024, kernel_size=kernel_size, padding='same', name='Convb6_3',
    #                   kernel_regularizer=regularizers.l2(norm_rate))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Convolution2D(1024, kernel_size=kernel_size, padding='same', name='Convb6_4',
    #                   kernel_regularizer=regularizers.l2(norm_rate))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size=pool_size, strides=stride, name='Pool6')(x)

    x = Flatten()(x)

    x = Dense(1024, kernel_regularizer=regularizers.l2(norm_rate), name='fc1')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1024, kernel_regularizer=regularizers.l2(norm_rate), name='fc2')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(num_class, activation='softmax', name='prediction',
                        kernel_regularizer=regularizers.l2(norm_rate))(x)
    # predictions = BatchNormalization()(x)
    model = Model(inputs=inputs, outputs=predictions)
    return model


if __name__ == '__main__':
    from keras.optimizers import Adam

    model = VGG_BN(205, norm_rate=0.0)
    print("DONE.")
    optimizer = Adam(1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  #
    model.summary()
