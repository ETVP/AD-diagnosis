from keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
img_file = r"F:\AIBL\TestNpy"
model = load_model(r"F:\SHold\code\pure\pure.ADvsNC.loss.b10.t3.h5")
model.summary()
predict_result = []
sigmoid = True
num = 0
files = os.listdir(img_file)


def two_classify_predict():
    """
    sigmoid 
    :return: 
    """
    labels = []
    for file in files:
        labels.append(file[0:2])
    labels = LabelEncoder().fit_transform(labels)
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
        if predict[num][0] <= 0.5 and labels[i] == 0:
            true_num += 1
        elif predict[num][0] > 0.5 and labels[i] == 1:
            true_num += 1
        else:
            false_num += 1
            print(file)

        i += 1

    print("True: ", true_num)
    print("False: ", false_num)
    print(true_num/(false_num + true_num))


if __name__ == '__main__':
    if sigmoid:
        two_classify_predict()



