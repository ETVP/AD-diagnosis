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
        embedded_layer = layers.SimpleRNN(output_size)(embedded_layer)
        self.model = embedded_layer

        # add sex
        sex_input = layers.Input(shape=(1, ), dtype='int32')
        embedded_layer = layers.Embedding(input_size, output_size)(sex_input)
        embedded_layer = layers.SimpleRNN(output_size)(embedded_layer)
        self.model = layers.concatenate([self.model, embedded_layer])

        # add age
        age_input = layers.Input(shape=(1, ), dtype='int32')
        embedded_layer = layers.Embedding(input_size, output_size)(age_input)
        embedded_layer = layers.SimpleRNN(output_size)(embedded_layer)
        self.model = layers.concatenate([self.model, embedded_layer])

        # add marriage
        marriage_input = layers.Input(shape=(1, ), dtype='int32')
        embedded_layer = layers.Embedding(input_size, output_size)(marriage_input)
        embedded_layer = layers.SimpleRNN(output_size)(embedded_layer)
        self.model = layers.concatenate([self.model, embedded_layer])

        # add apoe4
        apoe4_input = layers.Input(shape=(1, ), dtype='int32')
        embedded_layer = layers.Embedding(input_size, output_size)(apoe4_input)
        embedded_layer = layers.SimpleRNN(output_size)(embedded_layer)
        self.model = layers.concatenate([self.model, embedded_layer])

        # add education
        edu_input = layers.Input(shape=(1, ), dtype='int32')
        embedded_layer = layers.Embedding(input_size, output_size)(edu_input)
        embedded_layer = layers.SimpleRNN(output_size)(embedded_layer)
        self.model = layers.concatenate([self.model, embedded_layer])

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
