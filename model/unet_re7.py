import keras
from keras import layers
from keras import regularizers
from keras.utils import plot_model


class Unet:
    def __init__(self, shape):
        """
        将dropout输出的结果直接影像最终结果
        :param shape:
        """
        self.re_rate = 0.6
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

        self.f_block2 = layers.Conv3D(16, (3, 3, 3), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.mp2)
        self.f_block2 = layers.BatchNormalization()(self.f_block2)

        self.b_back2 = layers.Conv3D(32, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(self.f_block2)
        self.b_back2 = layers.BatchNormalization()(self.b_back2)

        self.b_back2 = layers.Conv3D(64, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(layers.UpSampling3D((2, 2, 2))(self.f_block2))
        self.b_back2 = layers.BatchNormalization()(self.b_back2)

        self.cat2 = layers.concatenate([self.f_block1, self.b_back2])
        self.bn = layers.BatchNormalization()(self.cat2)

        self.b_back1 = layers.Conv3D(32, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(layers.UpSampling3D((2, 2, 2))(self.bn))
        self.b_back1 = layers.BatchNormalization()(self.b_back1)

        self.gb = layers.GlobalAveragePooling3D()(self.b_back1)
        self.gb_drop = layers.Dropout(rate=0.9)(self.gb)

        self.pure_dense = layers.Dense(1, activation='sigmoid')(self.gb_drop)

        # add mmse
        mmse_input = layers.Input(shape=(1,), dtype='int32')
        embedded_layer = layers.Embedding(shape[-1], 1)(mmse_input)
        embedded_layer = layers.Conv1D(4, 1, activation='relu')(embedded_layer)
        embedded_layer = layers.Conv1D(4, 1, activation='relu')(embedded_layer)
        emCon = layers.Flatten()(embedded_layer)
        self.mmse = emCon

        # add sex
        sex_input = layers.Input(shape=(1,), dtype='int32')
        embedded_layer = layers.Embedding(shape[-1], 1)(sex_input)
        embedded_layer = layers.Conv1D(4, 1, activation='relu')(embedded_layer)
        embedded_layer = layers.Conv1D(4, 1, activation='relu')(embedded_layer)
        emCon = layers.Flatten()(embedded_layer)
        self.sex = emCon

        # add age
        age_input = layers.Input(shape=(1,), dtype='int32')
        embedded_layer = layers.Embedding(shape[-1], 1)(age_input)
        embedded_layer = layers.Conv1D(4, 1, activation='relu')(embedded_layer)
        embedded_layer = layers.Conv1D(4, 1, activation='relu')(embedded_layer)
        emCon = layers.Flatten()(embedded_layer)
        self.age = emCon

        # add marriage
        marriage_input = layers.Input(shape=(1,), dtype='int32')
        embedded_layer = layers.Embedding(shape[-1], 1)(marriage_input)
        embedded_layer = layers.Conv1D(4, 1, activation='relu')(embedded_layer)
        embedded_layer = layers.Conv1D(4, 1, activation='relu')(embedded_layer)
        emCon = layers.Flatten()(embedded_layer)
        self.marriage = emCon

        # add apoe4
        apoe4_input = layers.Input(shape=(1,), dtype='int32')
        embedded_layer = layers.Embedding(shape[-1], 1)(apoe4_input)
        embedded_layer = layers.Conv1D(4, 1, activation='relu')(embedded_layer)
        embedded_layer = layers.Conv1D(4, 1, activation='relu')(embedded_layer)
        emCon = layers.Flatten()(embedded_layer)
        self.apoe4 = emCon

        # add education
        edu_input = layers.Input(shape=(1,), dtype='int32')
        embedded_layer = layers.Embedding(shape[-1], 1)(edu_input)
        embedded_layer = layers.Conv1D(4, 1, activation='relu')(embedded_layer)
        embedded_layer = layers.Conv1D(4, 1, activation='relu')(embedded_layer)
        emCon = layers.Flatten()(embedded_layer)
        self.edu = emCon

        self.cat = layers.concatenate([self.mmse, self.sex, self.age, self.marriage, self.apoe4, self.edu])

        self.cli_dense = layers.Dense(6, activation='relu')(self.cat)

        self.cli_judge = layers.Dense(1, activation='sigmoid')(self.cli_dense)

        self.cat = layers.concatenate([self.gb_drop, self.cli_dense])

        self.dense = layers.Dense(1, activation='sigmoid')(self.cat)

        self.model = keras.Model(input=[self.inputs, mmse_input, sex_input, age_input, marriage_input,
                                        apoe4_input, edu_input], output=[self.pure_dense, self.cli_judge, self.dense])

    def get_model(self):
        return self.model


if __name__ == '__main__':
    model = Unet((64, 104, 80, 1)).get_model()
    model.summary()
    plot_model(model, "../img/unet_re7.png", True)
