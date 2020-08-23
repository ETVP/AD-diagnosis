import keras
from keras import layers
from keras import regularizers
from keras.utils import plot_model


class Resnet:
    def __init__(self, shape):
        self.re_rate = 0.9

        self.input = layers.Input(shape=shape)
        print(self.input.shape)

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
                                      padding= 'same')(self.b_block1)
        self.b_block2 = layers.BatchNormalization()(self.b_block2)

        # self.c_block1 = layers.concatenate([self.b_block0, self.b_block2])
        self.add1 = layers.add([self.b_block0, self.b_block2])
        self.add1 = layers.BatchNormalization()(self.add1)

        self.b_block2 = layers.Conv3D(16, (2, 2, 2), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.add1)

        self.b_block3 = layers.MaxPooling3D((2, 2, 2))(self.b_block2)

        self.b_block3 = layers.BatchNormalization()(self.b_block3)

        self.b_block4 = layers.Conv3D(16, (2, 2, 2), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.b_block3)
        self.b_block4 = layers.BatchNormalization()(self.b_block4)

        self.b_block5 = layers.Conv3D(16, (2, 2, 2), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.b_block4)
        self.b_block5 = layers.BatchNormalization()(self.b_block5)

        self.add2 = layers.add([self.b_block5, self.b_block3])
        self.add2 = layers.BatchNormalization()(self.add2)

        self.b_block6 = layers.Conv3D(32, (2, 2, 2), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.add2)
        self.b_block6 = layers.MaxPooling3D((2, 2, 2))(self.b_block6)
        self.b_block6 = layers.BatchNormalization()(self.b_block6)

        self.b_block7 = layers.Conv3D(32, (2, 2, 2), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.b_block6)
        self.b_block7 = layers.BatchNormalization()(self.b_block7)

        self.b_block8 = layers.Conv3D(32, (2, 2, 2), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.b_block7)
        self.b_block8 = layers.BatchNormalization()(self.b_block8)

        self.add3 = layers.add([self.b_block8, self.b_block6])
        self.add3 = layers.BatchNormalization()(self.add3)

        self.b_block9 = layers.Conv3D(64, (2, 2, 2), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.add3)
        self.b_block9 = layers.MaxPooling3D((2, 2, 2))(self.b_block9)
        self.b_block9 = layers.BatchNormalization()(self.b_block9)
        #
        self.b_block10 = layers.Conv3D(32, (2, 2, 2), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.b_block9)
        # self.b_block10 = layers.BatchNormalization()(self.b_block10)
        #
        # self.b_block11 = layers.Conv3D(64, (2, 2, 2), activation='relu',
        #                               kernel_regularizer=regularizers.l2(self.re_rate),
        #                               padding='same')(self.b_block10)
        # self.b_block11 = layers.BatchNormalization()(self.b_block11)
        #
        # self.add4 = layers.add([self.b_block11, self.b_block9])
        #
        # self.b_block12 = layers.Conv3D(64, (2, 2, 2), activation='relu',
        #                               kernel_regularizer=regularizers.l2(self.re_rate),
        #                               padding='same')(self.add4)
        # self.b_block12 = layers.MaxPooling3D((2, 2, 2))(self.b_block9)
        self.b_block12 = layers.BatchNormalization()(self.b_block10)

        self.drop = layers.Dropout(rate=0.8)(self.b_block12)

        self.transform = layers.Flatten()(self.drop)
        self.dense1 = layers.Dense(64, activation='relu')(self.transform)
        # self.dense1 = layers.Dropout(rate=0.2)(self.dense1)
        self.dense2 = layers.Dense(16, activation='relu')(self.dense1)
        # five classify begin
        # self.dense3 = layers.Dense(3, activation='softmax')(self.dense2)
        # five classify end
        # two classify begin
        self.dense3 = layers.Dense(1, activation='sigmoid')(self.dense2)
        # two classify end
        self.model = keras.Model(input=self.input, output=self.dense3)

    def get_model(self):
        return self.model


if __name__ == '__main__':
    model = Resnet((64, 104, 80, 1)).get_model()
    # plot_model(model, "img/model.png", True)
    # model.summary()
