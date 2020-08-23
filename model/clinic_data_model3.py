import keras
from keras import layers
from keras.utils import plot_model


class Unet:
    # delete one concatenate
    def __init__(self):
        self.re_rate = 0.9
        input_size = 4
        output_size = 1
        # add mmse
        mmse_input = layers.Input(shape=(1, ), dtype='int32')
        embedded_layer = layers.Embedding(input_size, output_size)(mmse_input)
        emCon = layers.Flatten()(embedded_layer)
        self.mmse = emCon

        # add sex
        sex_input = layers.Input(shape=(1, ), dtype='int32')
        embedded_layer = layers.Embedding(input_size, output_size)(sex_input)
        emCon = layers.Flatten()(embedded_layer)
        self.sex = emCon

        # add age
        age_input = layers.Input(shape=(1, ), dtype='int32')
        embedded_layer = layers.Embedding(input_size, output_size)(age_input)
        emCon = layers.Flatten()(embedded_layer)
        self.age = emCon

        # add marriage
        marriage_input = layers.Input(shape=(1, ), dtype='int32')
        embedded_layer = layers.Embedding(input_size, output_size)(marriage_input)
        emCon = layers.Flatten()(embedded_layer)
        self.marriage = emCon

        # add apoe4
        apoe4_input = layers.Input(shape=(1, ), dtype='int32')
        embedded_layer = layers.Embedding(input_size, output_size)(apoe4_input)
        emCon = layers.Flatten()(embedded_layer)
        self.apoe4 = emCon

        # add education
        edu_input = layers.Input(shape=(1, ), dtype='int32')
        embedded_layer = layers.Embedding(input_size, output_size)(edu_input)
        emCon = layers.Flatten()(embedded_layer)
        self.edu = emCon

        self.model = layers.concatenate([self.mmse, self.sex, self.age, self.marriage, self.apoe4, self.edu])

        self.model = layers.Dense(4, activation='relu')(self.model)
        self.model = layers.Dense(16, activation='relu')(self.model)
        self.model = layers.Dense(32, activation='relu')(self.model)
        self.model = layers.Dense(64, activation='relu')(self.model)

        self.model = layers.Dense(4, activation='softmax')(self.model)
        self.model = keras.Model(input=[mmse_input, sex_input, age_input, marriage_input,
                                        apoe4_input, edu_input], output=self.model)

    def get_model(self):
        return self.model


if __name__ == '__main__':
    model = Unet().get_model()
    model.summary()
    plot_model(model, "../img/unet1.png", True)
