from keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
img_file = r"/openbayes/home/data/train"
model = load_model("logs/unet.acc.b5.h5")
model.summary()
predict_result = []
sigmoid = True
num = 0
files = os.listdir(img_file)
true_file = open("true", "a+")


def two_classify_predict():
    """
    sigmoid
    :return:
    """
    labels = []

    tmp_files = []
    global files
    for f in files:
        if f.__contains__(".num.0."):
            tmp_files.append(f)
    files = tmp_files
    for file in files:
        labels.append(file[0:2])
    labels = LabelEncoder().fit_transform(labels)
    true_num = 0
    false_num = 0
    i = 0

    for file in files:
        img_data = np.load(os.path.join(img_file, file))
        img_data = img_data[np.newaxis, :, :, :, np.newaxis]
        predict = model.predict(img_data)
        print(predict, " ", labels[i])
        if predict[num][0] <= 0.5 and labels[i] == 0:
            true_num += 1
        elif predict[num][0] > 0.5 and labels[i] == 1:
            true_num += 1
        else:
            print(file)
            false_num += 1
        i += 1

    print("True: ", true_num)
    print("False: ", false_num)
    print(true_num/(false_num + true_num))

def softmax_two():
    labels = []
    for file in files:
        labels.append(file[0:2])
    labels = LabelEncoder().fit_transform(labels)
    labels = np_utils.to_categorical(labels)
    true_num = 0
    false_num = 0
    i = 0

    for file in files:
        img_data = np.load(os.path.join(img_file, file))
        img_data = np.squeeze(img_data)
        img_data = img_data[np.newaxis, :, :, :, np.newaxis]
        max_ = np.max(img_data)
        min_ = np.min(img_data)
        img_data = (img_data - min_) / (max_ - min_)
        predict = model.predict(img_data)
        print(predict, " ", labels[i])
        if predict[num][0] >= predict[num][1] and labels[i][0] == 1:
            true_num += 1
        elif predict[num][0] <= predict[num][1] and labels[i][1] == 1:
            true_num += 1
        else:
            print(file)
            false_num += 1
        i += 1

    print("True: ", true_num)
    print("False: ", false_num)
    print(true_num / (false_num + true_num))


if __name__ == '__main__':
    if sigmoid:
        two_classify_predict()
    else:
        softmax_two()

