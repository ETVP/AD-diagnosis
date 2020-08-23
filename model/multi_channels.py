import keras
from keras import layers
from keras import regularizers
from keras.utils import plot_model


class Model:
    def __init__(self, shapes):
        self.re_rate = 0.9
        self.dr = 0.9
        self.first_input = layers.Input(shape=shapes[0])
        self.first_gp = self.single_channel(self.first_input)

        self.second_input = layers.Input(shape=shapes[1])
        self.second_gp = self.single_channel(self.second_input)

        self.third_input = layers.Input(shape=shapes[2])
        self.third_gp = self.single_channel(self.third_input)

        self.forth_input = layers.Input(shape=shapes[3])
        self.forth_gp = self.single_channel(self.forth_input)

        self.fifth_input = layers.Input(shape=shapes[4])
        self.fifth_gp = self.single_channel(self.fifth_input)

        self.sixth_input = layers.Input(shape=shapes[5])
        self.sixth_gp = self.single_channel(self.sixth_input)

        self.cat = layers.concatenate([self.first_gp, self.second_gp, self.third_gp,
                                       self.forth_gp, self.fifth_gp, self.sixth_gp])

        self.dn = layers.Dense(1, activation='sigmoid')(self.cat)
        self.model = keras.Model(input=[self.first_input, self.second_input, self.third_input,
                                        self.forth_input, self.fifth_input, self.sixth_input], output=self.dn)

    def single_channel(self, input):
        conv1 = layers.Conv3D(4, (5, 6, 5), activation="relu",
                                   kernel_regularizer=regularizers.l2(self.re_rate))(input)
        conv2 = layers.Conv3D(8, (3, 3, 3), activation='relu',
                                   kernel_regularizer=regularizers.l2(self.re_rate))(conv1)
        mp_1 = layers.MaxPooling3D((2, 2, 2))(conv2)

        conv3 = layers.Conv3D(16, (3, 3, 3), activation='relu',
                                   kernel_regularizer=regularizers.l2(self.re_rate))(mp_1)
        conv4 = layers.Conv3D(16, (3, 3, 3), activation='relu',
                                   kernel_regularizer=regularizers.l2(self.re_rate))(conv3)

        mp_2 = layers.MaxPooling3D((2, 2, 2))(conv4)

        conv5 = layers.Conv3D(16, (3, 1, 3), activation='relu',
                                   kernel_regularizer=regularizers.l2(self.re_rate))(mp_2)
        conv6 = layers.Conv3D(16, (3, 1, 3), activation='relu',
                              kernel_regularizer=regularizers.l2(self.re_rate))(conv5)
        gb = layers.GlobalAveragePooling3D()(conv6)
        dr = layers.Dropout(rate=self.dr)(gb)
        return dr

    def get_model(self):
        return self.model


if __name__ == '__main__':
    model = Model([(50, 31, 50, 1) for i in range(6)]).get_model()
    plot_model(model, "./model.png", True)