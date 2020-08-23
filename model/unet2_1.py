import keras
from keras import layers
from keras import regularizers
from keras.utils import plot_model


class Unet:
    def __init__(self, shape):
        self.re_rate = 0.9
        self.inputs = layers.Input(shape=shape)
        ac = layers.LeakyReLU(alpha=0.3)
        self.f_block = layers.Conv3D(4, (5, 5, 5), activation=ac,
                                         kernel_regularizer=regularizers.l2(self.re_rate),
                                         padding='same')(self.inputs)
        self.bn = layers.BatchNormalization()(self.f_block)
        self.mp1 = layers.MaxPooling3D((2, 2, 2))(self.bn)

        self.f_block1 = layers.Conv3D(8, (5, 5, 5), activation=ac,
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.mp1)
        self.bn = layers.BatchNormalization()(self.f_block1)
        self.mp2 = layers.MaxPooling3D((2, 2, 2))(self.bn)

        self.f_block2 = layers.Conv3D(16, (4, 4, 4), activation=ac,
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.mp2)

        self.f_block2 = layers.BatchNormalization()(self.f_block2)

        self.mp3 = layers.MaxPooling3D((2, 2, 2))(self.f_block2)

        self.f_block3 = layers.Conv3D(16, (3, 3, 3), activation=ac,
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.mp3)
        self.f_block3 = layers.BatchNormalization()(self.f_block3)

        self.b_back3 = layers.Conv3D(16, (3, 3, 3), activation=ac,
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(self.f_block3)
        self.b_back3 = layers.BatchNormalization()(self.b_back3)

        self.cat1 = layers.concatenate([self.f_block3, self.b_back3])
        self.bn4 = layers.BatchNormalization()(self.cat1)

        self.b_back2 = layers.Conv3D(16, (3, 3, 3), activation=ac,
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(layers.UpSampling3D((2, 2, 2))(self.bn4))
        self.b_back2 = layers.BatchNormalization()(self.b_back2)
        self.cat2 = layers.concatenate([self.mp2, self.b_back2])

        self.b_back1 = layers.Conv3D(16, (3, 3, 3), activation=ac,
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(layers.UpSampling3D((2, 2, 2))(self.cat2))
        self.b_back1 = layers.BatchNormalization()(self.b_back1)

        self.cat3 = layers.concatenate([self.mp1, self.b_back1])

        self.gb = layers.GlobalAveragePooling3D()(self.cat3)
        # self.gb = layers.GlobalMaxPooling3D()(self.cat3)
        self.dense1 = layers.Dropout(rate=0.3)(self.gb)
        self.dense3 = layers.Dense(1, activation='sigmoid')(self.dense1)

        self.model = keras.Model(input=self.inputs, output=self.dense3)

    def get_model(self):
        return self.model


if __name__ == '__main__':
    model = Unet((64, 104, 80, 1)).get_model()
    model.summary()
    plot_model(model, "img/model.png", True)
