from keras import layers, models
from keras import regularizers
from keras.utils import plot_model


class Convnet:
    def __init__(self, shape):
        self.re_rate = 0.9
        self.model = models.Sequential()
        self.model.add(layers.Conv3D(16, (3, 3, 3),
                                     kernel_regularizer=regularizers.l2(self.re_rate), input_shape=shape))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())

        self.model.add(layers.Conv3D(16, (1, 1, 1),
                                     kernel_regularizer=regularizers.l2(self.re_rate)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.MaxPooling3D((2, 2, 2)))

        self.model.add(layers.Conv3D(32, (3, 3, 3),
                                     kernel_regularizer=regularizers.l2(self.re_rate)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())

        self.model.add(layers.Conv3D(32, (1, 1, 1),
                                     kernel_regularizer=regularizers.l2(self.re_rate)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.MaxPooling3D((2, 2, 2)))

        self.model.add(layers.Conv3D(64, (3, 3, 3),
                                     kernel_regularizer=regularizers.l2(self.re_rate)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.Conv3D(64, (1, 1, 1),
                                     kernel_regularizer=regularizers.l2(self.re_rate)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.MaxPooling3D((2, 2, 2)))

        self.model.add(layers.Conv3D(128, (3, 3, 3),
                                     kernel_regularizer=regularizers.l2(self.re_rate)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        # fifth change
        self.model.add(layers.Conv3D(128, (1, 1, 1),
                                     kernel_regularizer=regularizers.l2(self.re_rate)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        # end
        self.model.add(layers.MaxPooling3D((2, 2, 2)))

        self.model.add(layers.Flatten())

        self.model.add(layers.Dropout(rate=0.7))

        self.model.add(layers.Dense(128, activation='relu'))
        # one change add dropout rate = 0.3 can't overfit
        # two change change rate from 0.3 to 0.2, can't overfit but the train set's
        # acc is close to val set
        self.model.add(layers.Dropout(rate=0.6))
        # end
        self.model.add(layers.Dense(64, activation='relu'))
        # self.model.add(layers.Dense(8, activation='relu'))
        self.model.add(layers.Dense(4, activation='softmax'))
        # self.model.add(layers.Dense(1, activation='sigmoid'))

    def get_model(self):
        return self.model


if __name__ == '__main__':
    model = Convnet((88, 128, 128, 1)).get_model()
    model.summary()
    plot_model(model, "img/model.png", True)
