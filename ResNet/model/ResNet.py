import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Add
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, SparseCategoricalAccuracy

def ResBlock(x, filter_num, verbose=False):

    conv1 = Conv2D(filter_num, (3, 3), strides=(2, 2), padding='same')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation="ReLU")(conv1)

    conv2 = Conv2D(filter_num, (3, 3), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation="ReLU")(conv2)

    if verbose:
       x = Conv2D(filter_num*2, (3, 3), strides=(2, 2), padding='valid')(x)

    skip = Add()(conv2, x)
    skip = Activation(activation="ReLU")(skip)

    return skip

class ResNet_34:

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_net(self):
        input = keras.Input(shape=self.input_shape)

        # block1(head)
        conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(activation="ReLU")(conv1)
        conv1 = MaxPooling2D((3, 3), strides=(2, 2))(conv1)

        # block2
        RB1 = ResBlock(conv1, 64)
        RB1 = ResBlock(RB1, 64)
        RB1 = ResBlock(RB1, 64)

       # block3
        RB2 = ResBlock(RB1, 128, verbose=True)
        RB2 = ResBlock(RB2, 128)
        RB2 = ResBlock(RB2, 128)
        RB2 = ResBlock(RB2, 128)

        # block4

        RB3 = ResBlock(RB2, 256, verbose=True)
        RB3 = ResBlock(RB3, 256)
        RB3 = ResBlock(RB3, 256)
        RB3 = ResBlock(RB3, 256)
        RB3 = ResBlock(RB3, 256)
        RB3 = ResBlock(RB3, 256)

        # block5
        RB4 = ResBlock(RB3, 512, verbose=True)
        RB4 = ResBlock(RB4, 512)
        RB4 = ResBlock(RB4, 512)

        # Fully connected final layer
        fc = GlobalAveragePooling2D()(RB4)
        dense = Dense(1000, activation="SoftMax")(fc)
        output = Activation('SoftMax')(dense)

        model = keras.Model(inputs=input, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(0.002),
            loss=SparseCategoricalCrossentropy(),
            metrics=SparseCategoricalAccuracy()
        )

        return model


# if __name__ == "__main__":
#     img_rows, img_cols = 224, 224
#     input_shape = (1, 3, img_rows, img_cols)  # (batch, channels, height, width)
#     model = ResNet_34.build_net()
#     model.summary()
