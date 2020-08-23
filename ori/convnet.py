from keras import layers, models
from keras import regularizers
from keras.utils import plot_model


class Convnet:
    def __init__(self, shape):
        self.re_rate = 0.9
        self.model = models.Sequential()
        self.model.add(layers.Conv3D(16, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate), input_shape=shape))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv3D(16, (1, 1, 1), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling3D((2, 2, 2)))

        self.model.add(layers.Conv3D(32, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv3D(32, (1, 1, 1), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling3D((2, 2, 2)))

        self.model.add(layers.Conv3D(64, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv3D(64, (1, 1, 1), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling3D((2, 2, 2)))

        self.model.add(layers.Conv3D(128, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate)))
        self.model.add(layers.BatchNormalization())
        # fifth change
        self.model.add(layers.Conv3D(128, (1, 1, 1), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate)))
        self.model.add(layers.BatchNormalization())
        # end
        self.model.add(layers.MaxPooling3D((2, 2, 2)))

        self.model.add(layers.Flatten())

        # three rate from 0.8 to 0.6 and the first dense from 64 to 128,
        # while dropout rate between dense changed from 0.2 to 0.5
        # then, on one side, it can't overfit all train data, while the acc of val dataset
        # can be reduce when the train dataset's acc is too high

        # four change: the first dropout's rate from 0.6 to 0.7
        # the sencond Dense's kernel number from 32 to 64
        # overfit to train set but not up to 1 just 92%
        # fifth change: add conv and BN and change the first dropout's rate from 0.6 to 0.7
        # acc at train dataset is above 98%, while it's just 63% at val dataset
        # sixth: double these conv with kernel size =(1, 1, 1), and change the second dropout rate from 0.5 to 0.6
        # acc at val dataset is 64.4%
        # seventh: change the kernel size from (1, 1, 1) to (3, 3, 3) and add padding='same'
        # after change, this becomes come overfit even train dataset
        # thus change network to sixth and then double the conv kernel number and change the first
        # dropout rate from 0.7 to 0.8 and change the second rate from 0.6 to 0.7, find this can't overfit
        # change the second dropout rate from 0.7 to  0.6， can't overfit
        # next, change first dropout rate from 0.8 to 0.7 with the second dropout rate is 0.6

        self.model.add(layers.Dropout(rate=0.7))

        self.model.add(layers.Dense(128, activation='relu'))
        # one change add dropout rate = 0.3 can't overfit
        # two change change rate from 0.3 to 0.2, can't overfit but the train set's
        # acc is close to val set
        self.model.add(layers.Dropout(rate=0.6))
        # end
        self.model.add(layers.Dense(64, activation='relu'))
        # self.model.add(layers.Dense(8, activation='relu'))
        # self.model.add(layers.Dense(3, activation='softmax'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

    def get_model(self):
        return self.model


if __name__ == '__main__':
    model = Convnet((88, 128, 128, 1)).get_model()
    model.summary()
    plot_model(model, "img/model.png", True)
