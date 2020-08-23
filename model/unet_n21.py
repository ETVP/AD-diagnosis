import keras
from keras import layers
from keras import regularizers
from keras.utils import plot_model


class Unet:
    # delete one concatenate
    def __init__(self, shape):
        self.re_rate = 0.9
        self.inputs = layers.Input(shape=shape)

        self.f_block = layers.Conv3D(4, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(self.inputs)
        self.bn = layers.BatchNormalization()(self.f_block)
        self.mp1 = layers.MaxPooling3D((2, 2, 2))(self.bn)

        self.f_block1 = layers.Conv3D(8, (3, 3, 3), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.mp1)
        self.bn = layers.BatchNormalization()(self.f_block1)

        self.mp2 = layers.MaxPooling3D((2, 2, 2))(self.bn)

        self.f_block2 = layers.Conv3D(8, (3, 3, 3), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.mp2)
        self.f_block2 = layers.BatchNormalization()(self.f_block2)

        self.b_back2 = layers.Conv3D(8, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(self.f_block2)
        self.b_back2 = layers.BatchNormalization()(self.b_back2)

        self.b_back2 = layers.Conv3D(8, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(layers.UpSampling3D((2, 2, 2))(self.f_block2))
        self.b_back2 = layers.BatchNormalization()(self.b_back2)

        self.b_back1 = layers.Conv3D(8, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(layers.UpSampling3D((2, 2, 2))(self.b_back2))
        self.b_back1 = layers.BatchNormalization()(self.b_back1)

        self.gb = layers.GlobalAveragePooling3D()(self.b_back1)
        self.drop = layers.Dropout(rate=0.9)(self.gb)

        self.dense = layers.Dense(1, activation='sigmoid')(self.drop)
        self.model = keras.Model(input=[self.inputs], output=self.dense)

    def get_model(self):
        return self.model


if __name__ == '__main__':
    model = Unet((64, 104, 80, 1)).get_model()
    model.summary()
    plot_model(model, "../img/unet1.png", True)
