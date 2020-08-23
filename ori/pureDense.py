import keras
from keras import layers, models
from keras import regularizers
from keras.utils import plot_model


class Dense:
    def __init__(self, shape):
        self.re_rate = 0.9
        model = models.Sequential()
        model.add(layers.Conv3D(16, (3, 3, 3), activation='relu', input_shape=shape,
                                kernel_regularizer=regularizers.l2(self.re_rate)))
        model.add(layers.MaxPooling3D((6, 6, 6)))
        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(5, activation='softmax'))
        self.model = model

    def get_model(self):
        return self.model


if __name__ == '__main__':
    model = Dense((88, 128, 128, 1)).get_model()
    model.summary()

