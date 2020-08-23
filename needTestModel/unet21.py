import keras
from keras import layers
from keras import regularizers
from keras.utils import plot_model


class Unet:
    def __init__(self, shape):
        self.re_rate = 0.9
        dr = 0.9
        self.inputs = layers.Input(shape=shape)

        self.f_block = layers.Conv3D(4, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(self.inputs)
        self.mp1 = layers.MaxPooling3D((2, 2, 2))(self.f_block)
        self.bn1 = layers.BatchNormalization()(self.mp1)

        self.f_block1 = layers.Conv3D(16, (3, 3, 3), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.bn1)
        self.mp2 = layers.MaxPooling3D((2, 2, 2))(self.f_block1)
        self.bn2 = layers.BatchNormalization()(self.mp2)

        self.f_block2 = layers.Conv3D(32, (3, 3, 3), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.bn2)
        self.mp3 = layers.MaxPooling3D((2, 2, 2))(self.f_block2)
        self.bn3 = layers.BatchNormalization()(self.mp3)

        self.f_block3 = layers.Conv3D(64, (3, 3, 3), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.bn3)
        self.f_block3 = layers.BatchNormalization()(self.f_block3)

        self.b_back3 = layers.Conv3D(128, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(self.f_block3)
        self.b_back3 = layers.BatchNormalization()(self.b_back3)

        self.b_back2 = layers.Conv3D(64, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(layers.UpSampling3D((2, 2, 2))(self.b_back3))
        self.b_back2 = layers.BatchNormalization()(self.b_back2)

        self.cat2 = layers.concatenate([self.bn2, self.b_back2])

        self.b_back1 = layers.Conv3D(32, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(layers.UpSampling3D((2, 2, 2))(self.cat2))
        self.b_back1 = layers.BatchNormalization()(self.b_back1)

        self.cat3 = layers.concatenate([self.bn1, self.b_back1])

        self.gb = layers.GlobalAveragePooling3D()(self.cat3)
        self.dr = layers.Dropout(rate=dr)(self.gb)
        self.dense = layers.Dense(1, activation='sigmoid')(self.dr)

        self.model = keras.Model(input=self.inputs, output=self.dense)

    def get_model(self):
        return self.model


if __name__ == '__main__':
    model = Unet((64, 104, 80, 1)).get_model()
    model.summary()
    plot_model(model, r"C:\Users\fan\Desktop\model21.png", True)
