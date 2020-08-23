from keras import layers, models
from keras import regularizers
from keras.utils import plot_model


class Conv_cite:
    """
    3D CNN-based classification using sMRI and MD-DTI images for Alzheimer disease studies
    model likes
    """
    def __init__(self, shape):
        self.re_rate = 0.9
        self.model = models.Sequential()
        self.model.add(layers.Conv3D(16, (5, 5, 5), kernel_regularizer=regularizers.l2(self.re_rate),
                                     input_shape=shape))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.MaxPooling3D((2, 2, 2)))

        self.model.add(layers.Conv3D(32, (4, 4, 4), kernel_regularizer=regularizers.l2(self.re_rate),
                                     input_shape=shape))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.MaxPooling3D((2, 2, 2)))

        self.model.add(layers.Conv3D(64, (3, 3, 3), kernel_regularizer=regularizers.l2(self.re_rate),
                                     input_shape=shape))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.MaxPooling3D((2, 2, 2)))

        self.model.add(layers.Conv3D(128, (3, 3, 3), kernel_regularizer=regularizers.l2(self.re_rate),
                                     input_shape=shape))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.ReLU())
        self.model.add(layers.MaxPooling3D((2, 2, 2)))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(16))
        self.model.add(layers.Dense(8))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(2, activation='sigmoid'))

    def get_model(self):
        return self.model


if __name__ == '__main__':
    model = Conv_cite((64, 104, 80, 1)).get_model()
    plot_model(model, "/home/fan/Desktop/model.png")
    model.summary()
