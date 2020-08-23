from keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, auc, recall_score
from sklearn import metrics



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
img_file = r"/home/zhhfan/newSH1/test"
model = ""
# model = load_model("log/unet.loss.b5.h5")
# model.summary()
predict_result = []
sigmoid = True
num = 0
# files = os.listdir(img_file)
files = ""

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
    TP = TN = FP = FN = 0
    T2T = TT = 0  # 真的预测为真的的个数，真的的个数 sen = T2T / TT
    F2F = TF = 0  # 假的预测为假的的个数，假的的个数 spe = F2F / TF
    all_values = []  # 保存预测值和真实值，用于计算AUC

    for file in files:
        img_data = np.load(os.path.join(img_file, file))
        img_data = img_data[np.newaxis, :, :, :, np.newaxis]
        predict = model.predict(img_data)
        print(predict, " ", labels[i])
        all_values.append((predict[0][0], labels[i]))
        if predict[num][0] <= 0.5 and labels[i] == 0:
            TT += 1
            T2T += 1
            true_num += 1
            TP += 1
        elif predict[num][0] > 0.5 and labels[i] == 1:
            F2F += 1
            TF += 1
            true_num += 1
            TN += 1
        elif predict[num][0] <= 0.5 and labels[i] == 1:
            print(file)
            TF += 1
            false_num += 1
            FN += 1
        elif predict[num][0] > 0.5 and labels[i] == 0:
            print(file)
            TT += 1
            false_num += 1
            FP += 1
        i += 1

    print("True: ", true_num)
    print("False: ", false_num)
    print("Acc: ", true_num/(false_num + true_num))
    precision = TP / (TP + FP)
    print("precision: ", precision)
    recall = TP / (TP + FN)
    print("recall: ", recall)
    print("f1 score: ", 2*(precision * recall)/(precision + recall))
    print("SEN: ", T2T / TT)
    print("SPE: ", F2F / TF)
    roc_dots = roc_auc(all_values)
    print(roc_dots)


def roc_auc(values):
    # 对于标签为0的ROC曲线
    # https://www.jianshu.com/p/5df19746daf9
    tmp_values = []
    roc_dots = []
    test_ = []
    pre_ = []
    # for i in range(len(values)):
    #     pre_.append(1-values[i][0])
    #     test_.append(1-values[i][1])
        # pre_.append(values[i][0])
        # test_.append(values[i][1])

    for i in range(len(values)):
        test_.append(values[i][0])
        pre_.append(values[i][1])

    fpr, tpr, thresholds = metrics.roc_curve(test_, pre_)
    roc_auc = auc(fpr, tpr)  # auc为Roc曲线下的面积
    # 开始画ROC曲线
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.title('Receiver operating characteristic example')
    # plt.show()
    plt.savefig(r"C:\Users\fan\Desktop\roc.png")
    for i in range(len(fpr)):
        roc_dots.append((fpr[i], tpr[i]))
    return roc_dots


def plot_single_roc(dots):
    plt.figure()
    x_ = []
    y_ = []
    for dot in dots:
        x_.append(dot[0])
        y_.append(dot[1])
    plt.plot(x_, y_)
    plt.show()


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
        if predict[num][0] <= 0.5 and labels[i] == 0:
            true_num += 1
        elif predict[num][0] > 0.5 and labels[i] == 1:
            true_num += 1
        else:
            false_num += 1
        i += 1

    print("True: ", true_num)
    print("False: ", false_num)
    print(true_num / (false_num + true_num))


if __name__ == '__main__':
    # values = [(0.4864054, 0), (0.38390538, 0), (0.5970961, 0), (0.6008134, 1), (0.52496034, 1), (0.53963715, 1), (0.7714267, 1), (0.6604512, 1), (0.55207294, 0), (0.6735532, 1), (0.48533702, 0), (0.45497978, 0), (0.51014376, 1), (0.5559006, 1), (0.46939874, 0), (0.4309406, 0), (0.3808581, 0), (0.7339021, 1), (0.4742676, 0), (0.7457411, 1)]
    # plot_single_roc(roc_auc(values))
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    scores = np.array([0.3, 0.2, 0.7, 0.5, 0.4, 0.9, 0.6])
    values = []
    for i in range(7):
        values.append((y[i], scores[i]))
    roc_auc(values)

