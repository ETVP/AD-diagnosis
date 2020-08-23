from keras import layers, models
from keras import regularizers
# from keras.utils import plot_model


def get_model(x, y, z):
    rate = 0.3
    dropout_rate = 0.2
    model = models.Sequential()
    model.add(layers.Conv3D(4, (3, 3, 3), activation='relu', input_shape=(x, y, z, 1)))
    model.add(layers.Conv3D(8, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(rate)))
    model.add(layers.Conv3D(16, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(rate)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(rate)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(rate)))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(rate)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(rate)))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(rate)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(rate)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(rate)))
    model.add(layers.MaxPooling3D((2, 2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(rate)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_regularizer=regularizers.l2(rate)))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling3D())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    # plot_model(model, "./img/model.png", True)
    model.summary()
    return model


if __name__ == '__main__':
    get_model(85, 128, 128)
