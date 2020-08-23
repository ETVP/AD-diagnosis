import keras
from keras import layers
from keras import regularizers
from keras.utils import plot_model


class Resnet:
    def __init__(self, shape):
        self.re_rate = 0.9

        self.input = layers.Input(shape=shape)

        self.b_block0 = layers.Conv3D(8, (3, 3, 3), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.input)
        self.b_block0 = layers.BatchNormalization()(self.b_block0)
        self.b_block0 = layers.MaxPooling3D((2, 2, 2))(self.b_block0)

        self.b_block1 = layers.Conv3D(8, (3, 3, 3), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.b_block0)
        self.b_block1 = layers.BatchNormalization()(self.b_block1)

        self.b_block2 = layers.Conv3D(8, (3, 3, 3), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.b_block1)
        self.b_block2 = layers.BatchNormalization()(self.b_block2)

        self.add1 = layers.add([self.b_block0, self.b_block2])
        self.add1 = layers.BatchNormalization()(self.add1)

        self.b_block2 = layers.Conv3D(16, (3, 3, 3), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.add1)
        self.b_block2 = layers.BatchNormalization()(self.b_block2)
        self.b_block2 = layers.MaxPooling3D((2, 2, 2))(self.b_block2)

        self.b_block3 = layers.Conv3D(16, (3, 3, 3), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.b_block2)
        self.b_block3 = layers.BatchNormalization()(self.b_block3)

        self.b_block4 = layers.Conv3D(16, (3, 3, 3), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.b_block3)
        self.b_block4 = layers.BatchNormalization()(self.b_block4)

        self.add2 = layers.add([self.b_block4, self.b_block2])
        self.add2 = layers.BatchNormalization()(self.add2)

        self.b_block5 = layers.Conv3D(32, (3, 3, 3), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.add2)
        self.b_block5 = layers.MaxPooling3D((2, 2, 2))(self.b_block5)
        self.b_block5 = layers.BatchNormalization()(self.b_block5)

        self.b_block6 = layers.Conv3D(32, (3, 3, 3), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.b_block5)
        self.b_block6 = layers.BatchNormalization()(self.b_block6)

        self.b_block7 = layers.Conv3D(32, (3, 3, 3), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.b_block6)
        self.b_block7 = layers.BatchNormalization()(self.b_block7)

        self.add3 = layers.add([self.b_block7, self.b_block5])
        self.add3 = layers.BatchNormalization()(self.add3)

        # change: first dropout from 0.7 to 0.8 and second dropout rate from 0.4 to 0.5
        # change 0.5 to 0.6
        # from 0.6 to 0.8 can't overfit
        # change: change the second dropout to 0.7
        self.drop = layers.Dropout(rate=0.8)(self.add3)

        self.transform = layers.Flatten()(self.drop)
        self.dense1 = layers.Dense(64, activation='relu')(self.transform)
        self.dense1 = layers.Dropout(rate=0.5)(self.dense1)
        self.dense2 = layers.Dense(32, activation='relu')(self.dense1)
        # self.dense3 = layers.Dense(4, activation='softmax')(self.dense2)
        self.dense3 = layers.Dense(4, activation='softmax')(self.dense2)
        self.model = keras.Model(input=self.input, output=self.dense3)

    def get_model(self):
        return self.model


if __name__ == '__main__':
    model = Resnet((64, 104, 80, 1)).get_model()
    plot_model(model, "img/model.png", True)
    model.summary()
