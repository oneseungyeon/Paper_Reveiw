from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Add
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, SparseCategoricalAccuracy

class PlainNet :

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_net(self):

        inputs = keras.Input(shape=self.input_shape)

        # block1(head)
        conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(activation="ReLU")(conv1)
        conv1 = MaxPooling2D((3, 3), strides=(2, 2))(conv1)

        # block2
        # 1
        conv2 = Conv2D(64, (3, 3), padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(activation="ReLU")(conv2)
        # 2
        conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(activation="ReLU")(conv2)
        # 3
        conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(activation="ReLU")(conv2)
        # 4
        conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(activation="ReLU")(conv2)
        # 5
        conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(activation="ReLU")(conv2)
        # 6
        conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(activation="ReLU")(conv2)

        # block3
        # 1
        conv3 = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation(activation="ReLU")(conv3)
        # 2
        conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation(activation="ReLU")(conv3)
        # 3
        conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation(activation="ReLU")(conv3)
        # 4
        conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation(activation="ReLU")(conv3)
        # 5
        conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation(activation="ReLU")(conv3)
        # 6
        conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation(activation="ReLU")(conv3)

        # block4
        # 1
        conv4 = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(conv3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(activation="ReLU")(conv4)
        # 2
        conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(activation="ReLU")(conv4)
        # 3
        conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(activation="ReLU")(conv4)
        # 4
        conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(activation="ReLU")(conv4)
        # 5
        conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(activation="ReLU")(conv4)
        # 6
        conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(activation="ReLU")(conv4)
        # 7
        conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(activation="ReLU")(conv4)
        # 8
        conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(activation="ReLU")(conv4)
        # 9
        conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(activation="ReLU")(conv4)
        # 10
        conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(activation="ReLU")(conv4)
        # 11
        conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(activation="ReLU")(conv4)
        # 12
        conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(activation="ReLU")(conv4)

        # block5
        # 1
        conv5 = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(conv4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation(activation="ReLU")(conv5)
        # 2
        conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation(activation="ReLU")(conv5)
        # 3
        conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation(activation="ReLU")(conv5)
        # 4
        conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation(activation="ReLU")(conv5)
        # 5
        conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation(activation="ReLU")(conv5)
        # 6
        conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation(activation="ReLU")(conv5)

        # Fully connected final layer
        fc = GlobalAveragePooling2D()(conv5)
        dense = Dense(1000, activation="SoftMax")(fc)
        output = Activation('SoftMax')(dense)

        model = keras.Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(0.002),
            loss=SparseCategoricalCrossentropy(),
            metrics=SparseCategoricalAccuracy()
        )

        return model

# if __name__ == "__main__" :
#
#     img_rows, img_cols = 224, 224
#     input_shape = (1, 3, img_rows, img_cols)  # (batch, channels, height, width)
#     model = PlainNet.build_net()
#     model.summary()

