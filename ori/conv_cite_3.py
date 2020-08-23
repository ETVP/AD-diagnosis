from keras import layers, models
from keras import regularizers


class Conv_cite_3:
    """
    A Novel Multimodal MRI Analysis for Alzheimer’s Disease Based on
    Convolutional Neural Network
    """
    def __init__(self, shape):
        self.re_rate = 0.9
        self.model = models.Sequential()
        self.model.add(layers.Conv3D(16, (3, 3, 3), kernel_regularizer=regularizers.l2(self.re_rate),
                                     input_shape=shape))
        self.model.add(layers.ReLU())
        self.model.add(layers.Conv3D(16, (3, 3, 3), kernel_regularizer=regularizers.l2(self.re_rate)))
        self.model.add(layers.ReLU())
        self.model.add(layers.MaxPooling3D((2, 2, 2)))
        self.model.add(layers.Dropout(rate=0.25))

        self.model.add(layers.Conv3D(32, (3, 3, 3), kernel_regularizer=regularizers.l2(self.re_rate)))
        self.model.add(layers.ReLU())
        self.model.add(layers.Conv3D(32, (3, 3, 3), kernel_regularizer=regularizers.l2(self.re_rate)))
        self.model.add(layers.ReLU())
        self.model.add(layers.MaxPooling3D((2, 2, 2)))
        self.model.add(layers.Dropout(rate=0.25))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(16))
        self.model.add(layers.Dense(4, activation='softmax'))

    def get_model(self):
        return self.model


if __name__ == '__main__':
    model = Conv_cite_3((64, 104, 80, 1)).get_model()
    model.summary()
