from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Add
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, CategoricalAccuracy
from keras.utils import np_utils
from keras import backend as K

class VGG16:

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_net(self):

        input = keras.Input(shape=self.input_shape)

        # block1(head)
        conv1 = Conv2D(64, (3, 3), padding='same')(input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(activation="relu")(conv1)
        conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(activation="relu")(conv1)
        conv1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

        # block2
        conv2 = Conv2D(128, (3, 3), padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(activation="relu")(conv2)
        conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(activation="relu")(conv2)
        conv2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)

        # block3
        conv3 = Conv2D(256, (3, 3), padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation(activation="relu")(conv3)
        conv3 = Conv2D(256, (3, 3), padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation(activation="relu")(conv3)
        conv3 = Conv2D(256, (3, 3), padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation(activation="relu")(conv3)
        conv3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)

        # block4
        conv4 = Conv2D(512, (3, 3), padding='same')(conv3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(activation="relu")(conv4)
        conv4 = Conv2D(512, (3, 3), padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(activation="relu")(conv4)
        conv4 = Conv2D(512, (3, 3), padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(activation="relu")(conv4)
        conv4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)

        # block5
        conv5 = Conv2D(512, (3, 3), padding='same')(conv4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation(activation="relu")(conv5)
        conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation(activation="relu")(conv5)
        conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation(activation="relu")(conv5)
        conv5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5)

        # Fully connected final layer
        fc = Flatten()(conv5)
        dense = Dense(4096, activation="relu")(fc)
        dense = Dense(4096, activation="relu")(dense)
        dense = Dense(1000, activation="relu")(dense)
        output = Dense(self.num_classes, activation="softmax")(dense)

        #model
        model = keras.Model(inputs=input, outputs=output)
        model.compile(loss=SparseCategoricalCrossentropy(), optimizer=keras.optimizers.Adam(1e-31844), metrics=SparseCategoricalAccuracy())
        # model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(1e-31844),
        #               metrics=["accuracy"])

        return model


if __name__ == "__main__" :

    EPOCH_ITER = 100
    BATCH_SIZE = 100
    (x_train, y_train), (x_val, y_val) = keras.datasets.cifar10.load_data()

    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    # # one-hot encoding
    # y_train = np_utils.to_categorical(y_train)
    # y_val = np_utils.to_categorical(y_val)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')

    x_train /= 255
    x_val /= 255

    datagen = ImageDataGenerator()
    datagen.fit(x_train)

    VGG16 = VGG16(input_shape=(32, 32, 3), num_classes=10)
    model = VGG16.build_net()

    model.summary()
    callbacks = [
        # keras.callbacks.ModelCheckpoint(filepath='./resnet_weights/resnet20_{epoch:02d}.hdf5'),
        keras.callbacks.TensorBoard(log_dir="./logs",
                                    update_freq="batch")
    ]

    history = model.fit(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=len(x_train) / BATCH_SIZE,
                        epochs=EPOCH_ITER,
                        callbacks=[callbacks],
                        validation_data=(x_val, y_val))

    # score = model.evaluate(x_val, y_val, verbose=0)
    # print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')