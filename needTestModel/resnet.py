import keras
from keras import layers
from keras import regularizers
from keras.utils import plot_model


class Unet:
    def __init__(self, shape):
        self.re_rate = 0.9
        dr = 0.9
        self.inputs = layers.Input(shape=shape)
        self.block1 = layers.Conv3D(4, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(self.inputs)
        self.mp1 = layers.MaxPooling3D((2, 2, 2))(self.block1)
        self.bn1 = layers.BatchNormalization()(self.mp1)

        self.block2 = layers.Conv3D(32, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(self.bn1)
        self.bn2 = layers.BatchNormalization()(self.block2)

        self.block3 = layers.Conv3D(64, (3, 3, 3), activation='relu',
                                    kernel_regularizer=regularizers.l2(self.re_rate),
                                    padding='same')(self.bn2)
        self.bn3 = layers.BatchNormalization()(self.block3)

        self.block4 = layers.Conv3D(32, (3, 3, 3), activation='relu',
                                    kernel_regularizer=regularizers.l2(self.re_rate),
                                    padding='same')(self.bn3)
        self.bn4 = layers.BatchNormalization()(self.block4)

        self.add1 = layers.add([self.bn2, self.bn4])
        self.mp2 = layers.MaxPooling3D((2, 2, 2))(self.add1)

        self.block5 = layers.Conv3D(64, (3, 3, 3), activation='relu',
                                    kernel_regularizer=regularizers.l2(self.re_rate),
                                    padding='same')(self.mp2)
        self.bn5 = layers.BatchNormalization()(self.block5)

        self.block6 = layers.Conv3D(128, (3, 3, 3), activation='relu',
                                    kernel_regularizer=regularizers.l2(self.re_rate),
                                    padding='same')(self.bn5)
        self.bn6 = layers.BatchNormalization()(self.block6)

        self.block7 = layers.Conv3D(64, (3, 3, 3), activation='relu',
                                    kernel_regularizer=regularizers.l2(self.re_rate),
                                    padding='same')(self.bn6)
        self.bn7 = layers.BatchNormalization()(self.block7)

        self.add2 = layers.add([self.bn5, self.bn7])

        self.gb = layers.GlobalAveragePooling3D()(self.add2)
        self.dr = layers.Dropout(rate=dr)(self.gb)
        self.dense = layers.Dense(1, activation='sigmoid')(self.dr)

        self.model = keras.Model(input=self.inputs, output=self.dense)

    def get_model(self):
        return self.model


if __name__ == '__main__':
    model = Unet((64, 104, 80, 1)).get_model()
    model.summary()
    plot_model(model, r"C:\Users\fan\Desktop\model.png", True)
