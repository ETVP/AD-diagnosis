import keras
from keras import layers
from keras import regularizers
from keras.utils import plot_model



class Unet:
    def __init__(self, shape):
        self.re_rate = 0.9
        self.inputs = layers.Input(shape=shape)

        self.f_block = layers.Conv3D(4, (3, 3, 3), activation='relu',
                                         kernel_regularizer=regularizers.l2(self.re_rate),
                                         padding='same')(self.inputs)
        self.bn1 = layers.BatchNormalization()(self.f_block)
        self.mp1 = layers.MaxPooling3D((2, 2, 2))(self.bn1)

        self.f_block1 = layers.Conv3D(16, (3, 3, 3), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.mp1)
        # self.f_block1 = layers.Conv3D(32, (3, 3, 3), activation='relu',
        #                               kernel_regularizer=regularizers.l2(self.re_rate),
        #                               padding='same')(self.f_block1)
        self.bn2 = layers.BatchNormalization()(self.f_block1)
        self.mp2 = layers.MaxPooling3D((2, 2, 2))(self.f_block1)

        self.f_block2 = layers.Conv3D(16, (3, 3, 3), activation='relu',
                                      kernel_regularizer=regularizers.l2(self.re_rate),
                                      padding='same')(self.f_block1)
        self.f_block2 = layers.BatchNormalization()(self.f_block2)
        # self.f_block2 = layers.Conv3D(16, (3, 3, 3), activation='relu',
        #                               kernel_regularizer=regularizers.l2(self.re_rate),
        #                               padding='same')(self.f_block2)
        # self.bn3 = layers.BatchNormalization()(self.f_block2)
        self.mp3 = layers.MaxPooling3D((2, 2, 2))(self.f_block2)

        self.b_back3 = layers.Conv3D(16, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(layers.UpSampling3D((2, 2, 2))(self.mp3))
        self.b_back3 = layers.BatchNormalization()(self.b_back3)
        # self.b_back3 = layers.Conv3D(16, (3, 3, 3), activation='relu',
        #                              kernel_regularizer=regularizers.l2(self.re_rate),
        #                              padding='same')(self.b_back3)
        self.cat1 = layers.concatenate([self.b_back3, self.f_block2])
        self.bn4 = layers.BatchNormalization()(self.cat1)

        self.b_back4 = layers.Conv3D(16, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(self.bn4)
        self.b_back4 = layers.BatchNormalization()(self.b_back4)
        self.b_back4 = layers.Conv3D(16, (3, 3, 3), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(self.b_back4)
        self.b_back4 = layers.BatchNormalization()(self.b_back4)
        self.b_back4 = layers.concatenate([self.b_back4, self.f_block1])
        # two classify begin
        # self.b_back4 = layers.MaxPooling3D((2, 2, 2))(self.b_back4)
        # end
        self.bn5 = layers.BatchNormalization()(self.b_back4)
        self.b_back5 = layers.Conv3D(32, (2, 2, 2), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(self.bn5)
        self.b_back5 = layers.BatchNormalization()(self.b_back5)
        self.b_back5 = layers.MaxPooling3D((2, 2, 2))(self.b_back5)

        self.b_back6 = layers.Conv3D(32, (2, 2, 2), activation='relu',
                                     kernel_regularizer=regularizers.l2(self.re_rate),
                                     padding='same')(self.b_back5)
        self.b_back6 = layers.BatchNormalization()(self.b_back6)
        self.b_back6 = layers.MaxPooling3D((2, 2, 2))(self.b_back6)

        # self.drop = layers.Dropout(rate=0.5)(self.b_back6)
        self.transform = layers.Flatten()(self.b_back6)
        # five classify num=32
        self.dense1 = layers.Dense(32, activation='relu')(self.transform)
        self.dense1 = layers.Dropout(rate=0.2)(self.dense1)
        # self.dense1 = layers.Dropout(rate=0.5)(self.dense1)
        # five classify rate=0.3
        # self.dense1 = layers.Dropout(rate=0.3)(self.dense1)
        # self.dense1 = layers.Dense(16, activation='relu')(self.dense1)
        self.dense2 = layers.Dense(16, activation='relu')(self.dense1)
        # five classify begin
        # self.dense2 = layers.Dense(8, activation='relu')(self.dense2)
        # self.dense3 = layers.Dense(5, activation='softmax')(self.dense2)
        self.dense3 = layers.Dense(4, activation='softmax')(self.dense2)
        # five classify end
        # two classify begin
        # self.dense3 = layers.Dense(8, activation='relu')(self.dense2)
        # self.dense3 = layers.Dense(1, activation='sigmoid')(self.dense2)
        # two classify end
        self.model = keras.Model(input=self.inputs, output=self.dense3)

    def get_model(self):
        return self.model


if __name__ == '__main__':
    model = Unet((64, 104, 80, 1)).get_model()
    model.summary()
    plot_model(model, "img/model.png", True)
