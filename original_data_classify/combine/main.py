from keras.optimizers import SGD, Adam
import keras
from ori import fan_tools, gen
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(img_file, model, filepath, epoch, batch_size, sigmoid=False):
    train_file = os.path.join(img_file, "train")
    val_file = os.path.join(img_file, "val")
    train_gen = gen.Generater(train_file).generate_arrays_from_file(batch_size, sigmoid)
    val_gen = gen.Generater(val_file).generate_arrays_from_file(batch_size, sigmoid)

    model.summary()

    sgd = SGD(lr=0.001, decay=1 / 40, nesterov=True, momentum=0.9)
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # 保存最好的模型参数
    callbacks_list = [
        keras.callbacks.ModelCheckpoint(filepath=filepath[0], monitor='val_loss', save_best_only=True, ),
        keras.callbacks.ModelCheckpoint(filepath=filepath[1], monitor='val_acc', save_best_only=True, ),
        keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, cooldown=0,
                                          min_lr=0.000001, verbose=0, ),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, cooldown=0,
                                          min_lr=0.000001, verbose=0, ),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, verbose=2), ]

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    # two classify end

    history = model.fit_generator(train_gen, steps_per_epoch=len(os.listdir(train_file)) // batch_size,
                                  validation_data=val_gen, validation_steps=len(os.listdir(val_file)) // batch_size,
                                  epochs=epoch, callbacks=callbacks_list)
    model.save("logs/unet.b5.h5")
    fan_tools.draw_acc_and_loss_with_val(history, filepath[2])


