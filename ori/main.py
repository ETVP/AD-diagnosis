from keras.optimizers import SGD, Adam
import keras
from ori import fan_tools, gen
# from densenet import get_model
# from keras.utils import multi_gpu_model
# from keras.models import load_model
import os
# from unet import Unet
# # from unet_3d import Unet
# from convnet import Convnet
from ori.resnet1 import Resnet
# from pureDense import Dense


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(cli_data, model, filepath, epoch, batch_size, sigmoid=False):
    numbers = [i for i in range(len(cli_data))]
    train_numbers = numbers[:int(len(numbers)/9)]
    val_numbers = numbers[int(len(numbers)/9):]
    train_gen = gen.Generater(cli_data, train_numbers).generate_arrays_from_file(batch_size, sigmoid)
    val_gen = gen.Generater(cli_data, val_numbers).generate_arrays_from_file(batch_size, sigmoid)

    model.summary()

    sgd = SGD(lr=0.001, decay=1 / 40, nesterov=True, momentum=0.9)
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # 保存最好的模型参数
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(filepath=filepath[0], monitor='val_loss', save_best_only=True, ),
        keras.callbacks.ModelCheckpoint(filepath=filepath[1], monitor='val_acc', save_best_only=True, ),
        keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=20, cooldown=0,
                                          min_lr=0.000001, verbose=0, ),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, cooldown=0,
                                          min_lr=0.000001, verbose=0, ),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=2), ]

    # multi-classify begin
    # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # multi-classify end
    # two classify begin
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    # two classify end

    history = model.fit_generator(train_gen, steps_per_epoch=len(os.listdir(train_file)) // batch_size,
                                  validation_data=val_gen, validation_steps=len(os.listdir(val_file)) // batch_size,
                                  epochs=epoch, callbacks=callbacks_list)
    fan_tools.draw_acc_and_loss_with_val(history, filepath[2])


if __name__ == '__main__':
    img_file = "/home/fan/Desktop/processed/ADvsMCIvsNC"
    model = Resnet((64, 104, 80, 1)).get_model()
    loss_file = "log/resnet.loss.3.b5.h5"
    acc_file = "log/resnet.loss.3.b5.h5"
    draw_file = "resnet.3.b5."
    epoch = 1
    batch_size = 5
    train(img_file, model, [loss_file, acc_file, draw_file], epoch, batch_size)

