from keras import layers, models
from keras import regularizers
import keras
from keras.utils import plot_model


class Resnet_cite:
    """
    Visual Explanations From Deep 3D Convolutional Neural Networks for
    Alzheimer’s Disease Classification (right)
    """
    def __init__(self, shape):
        self.re_rate = 0.9
        self.input = layers.Input(shape=shape)

        self.modle = layers.Conv3D(32, (3, 3, 3), padding='same')(self.input)
        self.modle = layers.BatchNormalization()(self.modle)
        self.modle = layers.ReLU()(self.modle)

        self.modle = layers.Conv3D(32, (3, 3, 3), padding='same')(self.modle)
        self.modle = layers.BatchNormalization()(self.modle)
        self.modle = layers.ReLU()(self.modle)

        self.modle = layers.Conv3D(64, (3, 3, 3), padding='same', strides=2)(self.modle)

        self.tmp = layers.BatchNormalization()(self.modle)
        self.tmp = layers.ReLU()(self.tmp)
        self.tmp = layers.Conv3D(64, (3, 3, 3), padding='same')(self.tmp)
        self.tmp = layers.BatchNormalization()(self.tmp)
        self.tmp = layers.ReLU()(self.tmp)
        self.tmp = layers.Conv3D(64, (3, 3, 3), padding='same')(self.tmp)

        self.modle = layers.add([self.modle, self.tmp])

        self.tmp = layers.BatchNormalization()(self.modle)
        self.tmp = layers.ReLU()(self.tmp)
        self.tmp = layers.Conv3D(64, (3, 3, 3), padding='same')(self.tmp)
        self.tmp = layers.BatchNormalization()(self.tmp)
        self.tmp = layers.ReLU()(self.tmp)
        self.tmp = layers.Conv3D(64, (3, 3, 3), padding='same')(self.tmp)

        self.modle = layers.add([self.modle, self.tmp])

        self.modle = layers.BatchNormalization()(self.modle)
        self.modle = layers.ReLU()(self.modle)
        self.modle = layers.Conv3D(64, (3, 3, 3), padding='same', strides=2)(self.modle)

        self.tmp = layers.BatchNormalization()(self.modle)
        self.tmp = layers.ReLU()(self.tmp)
        self.tmp = layers.Conv3D(64, (3, 3, 3), padding='same')(self.tmp)
        self.tmp = layers.BatchNormalization()(self.tmp)
        self.tmp = layers.ReLU()(self.tmp)
        self.tmp = layers.Conv3D(64, (3, 3, 3), padding='same')(self.tmp)

        self.modle = layers.add([self.modle, self.tmp])

        self.tmp = layers.BatchNormalization()(self.modle)
        self.tmp = layers.ReLU()(self.tmp)
        self.tmp = layers.Conv3D(64, (3, 3, 3), padding='same')(self.tmp)
        self.tmp = layers.BatchNormalization()(self.tmp)
        self.tmp = layers.ReLU()(self.tmp)
        self.tmp = layers.Conv3D(64, (3, 3, 3), padding='same')(self.tmp)

        self.modle = layers.add([self.modle, self.tmp])

        self.modle = layers.BatchNormalization()(self.modle)
        self.modle = layers.ReLU()(self.modle)
        self.modle = layers.Conv3D(128, (3, 3, 3), padding='same', strides=2)(self.modle)

        self.tmp = layers.BatchNormalization()(self.modle)
        self.tmp = layers.ReLU()(self.tmp)
        self.tmp = layers.Conv3D(128, (3, 3, 3), padding='same')(self.tmp)
        self.tmp = layers.BatchNormalization()(self.tmp)
        self.tmp = layers.ReLU()(self.tmp)
        self.tmp = layers.Conv3D(128, (3, 3, 3), padding='same')(self.tmp)

        self.modle = layers.add([self.modle, self.tmp])

        self.tmp = layers.BatchNormalization()(self.modle)
        self.tmp = layers.ReLU()(self.tmp)
        self.tmp = layers.Conv3D(128, (3, 3, 3), padding='same')(self.tmp)
        self.tmp = layers.BatchNormalization()(self.tmp)
        self.tmp = layers.ReLU()(self.tmp)
        self.tmp = layers.Conv3D(128, (3, 3, 3), padding='same')(self.tmp)

        self.modle = layers.add([self.modle, self.tmp])

        self.modle = layers.GlobalAveragePooling3D()(self.modle)
        self.modle = layers.Dense(4, activation='softmax')(self.modle)

        self.modle = keras.Model(input=self.input, output=self.modle)

    def get_model(self):
        return self.modle


if __name__ == '__main__':
    model = Resnet_cite((64, 104, 80, 1)).get_model()
    model.summary()
    plot_model(model, "/home/fan/Desktop/resnet.png")
