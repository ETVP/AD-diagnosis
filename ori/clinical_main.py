from keras.optimizers import SGD, Adam
import keras
from ori import gen_clinical, fan_tools
import os
import random
from keras.models import load_model


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(model, filepath, epoch, batch_size, sigmoid=False, accmodel=""):

    random_nums = [i for i in range(723)]
    random.shuffle(random_nums)
    train_nums = random_nums[0: int(723/5*3)]
    val_nums = random_nums[int(723/5*3)+1:int(723/5*4)]
    test_nums = random_nums[int(723/5*4)+1:723]

    excel_path = r"C:\Users\fan\Desktop\AD\ADNCClinical\clinicalData.xlsx"
    train_gen = gen_clinical.Generater(excel_path, train_nums).generate_arrays_from_file(batch_size, sigmoid)
    val_gen = gen_clinical.Generater(excel_path, val_nums).generate_arrays_from_file(batch_size, sigmoid)
    test_gen = gen_clinical.Generater(excel_path, test_nums).generate_arrays_from_file(batch_size, sigmoid)

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

    # model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(train_gen, steps_per_epoch=434 // batch_size,
                                  validation_data=val_gen, validation_steps=145 // batch_size,
                                  epochs=epoch, callbacks=callbacks_list)
    fan_tools.draw_acc_and_loss_with_val(history, filepath[2])

    model = load_model(accmodel)

    pred = model.predict_generator(test_gen,
                            steps=145 // batch_size)
    score = model.evaluate_generator(test_gen, steps=145 // batch_size)
    print("loss ", score[0])
    print("Acc ", score[1])
    print(pred)
