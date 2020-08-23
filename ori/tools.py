import os
import matplotlib.pyplot as plt
import numpy as np


def two_sen(path, rate):
    """
    敏感性(SEN)
    sigmoid
    :param path:
    :return:
    """
    true_positive = 0
    positive = 0
    data = read_data(path)
    for tmp in data:
        tmp = tmp.split(" ")
        tmp[0] = tmp[0].replace("[", "").replace("]", "")
        tmp[-1] = tmp[-1].strip()
        pre = float(tmp[0])
        true_value = int(tmp[-1])
        if true_value == 1 and pre >= rate:
            positive += 1
            true_positive += 1
        elif true_value == 1:
            positive += 1
    return round(true_positive / positive, 2)


def two_spe(path, rate):
    """
    特异性(SPE)
    sigmoid
    :param path:
    :return:
    """
    non_positive = 0  # number in original data
    true_non_positive = 0  # predict and it's true
    data = read_data(path)
    for tmp in data:
        tmp = tmp.split(" ")
        tmp[0] = tmp[0].replace("[", "").replace("]", "")
        tmp[-1] = tmp[-1].strip()
        pre = float(tmp[0])
        true_value = int(tmp[-1])
        if true_value == 0 and pre < rate:
            non_positive += 1
            true_non_positive += 1
        elif true_value == 0:
            non_positive += 1
    return round(true_non_positive / non_positive, 2)


def two_ppv(path):
    """
    阳性预测值(postive predictive value -- PPV)
    :param path:
    :return:
    """
    data = read_data(path)
    true_positive = 0
    false_positive = 0
    for tmp in data:
        tmp = tmp.split(" ")
        tmp[0] = tmp[0].replace("[", "").replace("]", "")
        tmp[-1] = tmp[-1].strip()
        pre = float(tmp[0])
        true_value = int(tmp[-1])
        if true_value == 1 and pre >= 0.5:
            true_positive += 1
        elif pre >= 0.5:
            false_positive += 1
    return round(true_positive / (true_positive + false_positive), 2)


def two_npv(path):
    """
    阴性预测值（negative predictive value -- NPV）
    :param path:
    :return:
    """
    data = read_data(path)
    true_non_positive = 0
    false_non_positive = 0
    for tmp in data:
        tmp = tmp.split(" ")
        tmp[0] = tmp[0].replace("[", "").replace("]", "")
        tmp[-1] = tmp[-1].strip()
        pre = float(tmp[0])
        true_value = int(tmp[-1])
        if true_value == 0 and pre < 0.5:
            true_non_positive += 1
        elif pre < 0.5:
            false_non_positive += 1
    return round(true_non_positive / (true_non_positive + false_non_positive), 2)


def two_roc(path):
    """
    受试者工作特征曲线 （receiver operating characteristic curve，ROC）
    :param path:
    :return:
    """
    sens = []
    minus_spes = []
    for i in range(101):
        rate = round(0.01*i, 2)
        sens.append(two_sen(path, rate))
        minus_spes.append(1-two_spe(path, rate))

    # plt.figure()
    # plt.plot(minus_spes, sens)
    # plt.show()
    return sens, minus_spes


def multi_roc(minus_spes, sens, img_path):
    plt.figure()
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    print(minus_spes[0])
    print(sens[0])
    plt.plot(minus_spes[0], sens[0], color='orange', label='convnet ( '+str(auc(sens[0]))+" )")
    print(minus_spes[1])
    print(sens[1])
    plt.plot(minus_spes[1], sens[1], color='green', label='resnet ( '+str(auc(sens[1]))+" )")
    print(minus_spes[2])
    print(sens[2])
    plt.plot(minus_spes[2], sens[2], color='blue', label='unet ( '+str(auc(sens[2]))+" )")
    print(minus_spes[3])
    print(sens[3])
    plt.plot(minus_spes[3], sens[3], color='yellow', label='unet2 ( '+str(auc(sens[3]))+" )")
    print(minus_spes[4])
    print(sens[4])
    plt.plot(minus_spes[4], sens[4], color='red', label='conv_cite (' + str(auc(sens[4])) + " )")
    plt.legend()
    plt.savefig(img_path)
    plt.show()


def auc(sen):
    auc = 0
    for i in range(len(sen)-1):
        auc += 0.01 * (sen[i+1] + sen[i])/2
    return round(auc, 2)


def read_data(path):
    f = open(path, 'r')
    data_flag = False
    data = []
    while True:
        if not data_flag and f.readline().__contains__("Non-trainable params"):
            f.readline()
            data_flag = True

        if data_flag:
            current_line = f.readline()
            if current_line.__contains__("True"):
                break
            data.append(current_line)
    return data


def mean_variance(arr):
    arr = np.array(arr)
    mean = np.mean(arr)
    variance = np.var(arr)
    return round(mean, 2), round(variance, 2)


def draw_compare(arr, save_name, title):
    """
    交叉验证准确率
    :param arr:
    :param save_name:
    :param title:
    :return:
    """
    arr = np.array(arr)
    # conv_cite = arr[0]
    conv = arr[0]
    res = arr[1]
    unet2 = arr[2]
    unet = arr[3]
    x_data = [1, 2, 3, 4, 5]
    plt.figure()
    plt.title(title)
    plt.xlabel("number of cross validation")
    plt.ylabel("accuracy rate (%)")
    # mean, var = mean_variance(conv_cite)
    # plt.plot(x_data, conv_cite, color='orange', label='conv_cite ('+str(mean)+", "+str(var)+')')
    mean, var = mean_variance(conv)
    plt.plot(x_data, conv, color='green', label='conv ('+str(mean)+", "+str(var)+')')
    mean, var = mean_variance(res)
    plt.plot(x_data, res, color='blue', label='resnet ('+str(mean)+", "+str(var)+')')
    mean, var = mean_variance(unet2)
    plt.plot(x_data, unet2, color='yellow', label='unet2 ('+str(mean)+", "+str(var)+')')
    mean, var = mean_variance(unet)
    plt.plot(x_data, unet, color='red', label='unet ('+str(mean)+", "+str(var)+')')
    plt.legend()
    plt.savefig(os.path.join(pic_path, save_name))
    plt.show()


def draw_three_compare(arr, save_name, title):
    """
    交叉验证准确率
    :param arr:
    :param save_name:
    :param title:
    :return:
    """
    arr = np.array(arr)
    conv = arr[0]
    res = arr[1]
    unet2 = arr[3]
    unet = arr[2]
    x_data = [1, 2, 3, 4, 5]
    plt.figure()
    plt.title(title)
    plt.xlabel("number of cross validation")
    plt.ylabel("accuracy rate (%)")
    mean, var = mean_variance(conv)
    plt.plot(x_data, conv, color='green', label='conv ('+str(mean)+", "+str(var)+')')
    mean, var = mean_variance(res)
    plt.plot(x_data, res, color='blue', label='resnet ('+str(mean)+", "+str(var)+')')
    mean, var = mean_variance(unet2)
    plt.plot(x_data, unet2, color='yellow', label='unet2 ('+str(mean)+", "+str(var)+')')
    mean, var = mean_variance(unet)
    plt.plot(x_data, unet, color='red', label='unet ('+str(mean)+", "+str(var)+')')
    plt.legend()
    plt.savefig(os.path.join(pic_path, save_name))
    plt.show()


def draw_two_acc():
    # AD vs NC acc
    conv_cite = [93.18, 97.73, 95.65, 91.67, 86.95]
    conv = [97.73, 97.73, 95.65, 87.5, 84.78]
    res = [100, 95.65, 84.78, 87.5, 86.96]
    unet2 = [97.73, 100, 91.3, 82.54, 84.78]
    unet = [97.73, 100, 89.13, 89.58, 89.13]
    arr = [conv_cite, conv, res, unet2, unet]
    draw_compare(arr, 'ADvsNC.png', 'ADvsNC')

    # NC vs EM acc
    conv_cite = [86.36, 97.83, 91.0, 95.45, 77.27]
    conv = [95.45, 60.87, 86.96, 93.18, 65.91]
    res = [93.18, 73.91, 80.43, 93.18, 72.73]
    unet2 = [88.64, 71.74, 89.13, 93.18, 72.73]
    unet = [90.91, 80.43, 91.30, 95.45, 72.73]
    arr = [conv_cite, conv, res, unet2, unet]
    draw_compare(arr, 'NCvsEM.png', 'NCvsEM')

    # ADvsLM acc
    conv_cite = [93.18, 84.78, 93.48, 80.43, 95.83]
    conv = [80.43, 76.09, 86.96, 73.91, 81.25]
    res = [81.82, 80.43, 91.30, 60.87, 87.5]
    unet2 = [100, 82.61, 91.3, 67.39, 87.5]
    unet = [90.91, 82.61, 93.48, 67.38, 89.58]
    arr = [conv_cite, conv, res, unet2, unet]
    draw_compare(arr, 'ADvsLM.png', 'ADvsLM')

    # EMvsLM
    conv_cite = [93.18, 100, 93.48, 88.64, 95.45]
    conv = [86.36, 93.48, 84.78, 63.64, 97.73]
    res = [88.64, 95.65, 89.13, 75, 97.73]
    unet2 = [93.18, 97.83, 89.13, 65.90, 97.73]
    unet = [95.45, 100, 91.30, 84.09, 95.45]
    arr = [conv_cite, conv, res, unet2, unet]
    draw_compare(arr, 'EMvsLM.png', 'EMvsLM')


def draw_three_acc():
    conv = [90.91, 94.12, 87.14, 89.71, 78.57]
    resnet = [80.3, 85.29, 90.0, 88.24, 61.14]
    unet = [89.39, 88.24, 88.57, 89.71, 71.43]
    unet2 = [95.45, 92.65, 88.57, 83.82, 62.86]
    arr = [conv, resnet, unet, unet2]
    draw_three_compare(arr, "NCvsEMvsLM.png", "NC vs EM vs LM")


def draw_multi_param(sens, spes, ppvs, npvs, aucs, title):
    plt.figure()
    plt.title(title)
    x_data = [1, 2, 3, 4, 5]
    plt.xlabel('number of cross validation')

    flag = ['o', 's', '+', '*', 'p']
    type = ['conv_cite', 'conv', 'resnet', 'unet', 'unet2']
    colors = ['green', 'blue', 'yellow', 'red', 'purple']
    for i in range(5):
        plt.plot(x_data, sens[i], color=colors[i], label=type[i] + ' sen', marker=flag[0])
        plt.plot(x_data, spes[i], color=colors[i], label=type[i] + ' spe', marker=flag[1])
        plt.plot(x_data, ppvs[i], color=colors[i], label=type[i] + ' ppv', marker=flag[2])
        plt.plot(x_data, npvs[i], color=colors[i], label=type[i] + ' npv', marker=flag[3])
        plt.plot(x_data, aucs[i], color=colors[i], label=type[i] + ' auc', marker=flag[4])
    plt.legend()
    plt.show()


def draw_multi_roc():
    path1 = "/home/fan/Desktop/ad_classify/log/ADvsNC/T2/conv1/result.md"
    path2 = "/home/fan/Desktop/ad_classify/log/ADvsNC/T2/resnet/result.md"
    path3 = "/home/fan/Desktop/ad_classify/log/ADvsNC/T2/unet2/result.md"
    path4 = "/home/fan/Desktop/ad_classify/log/ADvsNC/T2/unet/result.md"  # u2
    path5 = "/home/fan/Desktop/ad_classify/log/ADvsNC/T2/conv/result.md"

    img_path = os.path.join(pic_path, "roc.png")
    paths = []
    paths.append(path1)
    paths.append(path2)
    paths.append(path3)
    paths.append(path4)
    paths.append(path5)

    sens = []
    minus_spes = []

    sen, minus_spe = two_roc(path1)
    sens.append(sen)
    minus_spes.append(minus_spe)

    sen, minus_spe = two_roc(path2)
    sens.append(sen)
    minus_spes.append(minus_spe)

    sen, minus_spe = two_roc(path3)
    sens.append(sen)
    minus_spes.append(minus_spe)

    sen, minus_spe = two_roc(path4)
    sens.append(sen)
    minus_spes.append(minus_spe)

    sen, minus_spe = two_roc(path5)
    sens.append(sen)
    minus_spes.append(minus_spe)

    multi_roc(minus_spes, sens, img_path)


def draw_multi_param_ADvsNC():
    # AD vs NC
    sens = [[1.0, 0.96, 0.96, 0.92, 0.92], [1.0, 0.96, 0.92, 0.92, 0.88], [1.0, 0.92, 0.92, 0.83, 0.79],
            [0.95, 1.0, 1.0, 0.92, 0.88], [0.95, 1.0, 0.92, 0.88, 0.92]]
    spes = [[0.86, 1.0, 0.95, 0.92, 0.82], [0.95, 1.0, 1.0, 0.83, 0.82], [1.0, 1.0, 0.77, 0.92, 0.95],
            [1.0, 1.0, 0.77, 0.88, 0.91], [1.0, 1.0, 0.91, 0.83, 0.77]]
    ppvs = [[0.88, 1.0, 0.96, 0.92, 0.85], [0.96, 1.0, 1.0, 0.85, 0.84], [1.0, 1.0, 0.81, 0.91, 0.95],
            [1.0, 1.0, 0.83, 0.88, 0.91], [1.0, 1.0, 0.92, 0.84, 0.81]]
    npvs = [[1.0, 0.96, 0.95, 0.92, 0.90], [1.0, 0.96, 0.92, 0.91, 0.86], [1.0, 0.92, 0.89, 0.85, 0.81],
            [0.96, 1.0, 1.0, 0.91, 0.87], [0.96, 1.0, 0.91, 0.87, 0.89]]
    aucs = [[0.99, 0.96, 0.95, 0.88, 0.76], [0.98, 0.95, 0.92, 0.88, 0.85], [0.99, 0.89, 0.91, 0.79, 0.76],
            [0.93, 0.81, 0.98, 0.88, 0.87], [0.91, 0.95, 0.89, 0.80, 0.89]]
    draw_multi_param(sens, spes, ppvs, npvs, aucs, 'AD vs NC')


def draw_multi_auc(aucs, title, png_name):
    plt.figure()
    plt.title(title)
    x_data = [1, 2, 3, 4, 5]
    plt.xlabel('number of cross validation')
    plt.ylabel('auc')
    colors = ['green', 'blue', 'yellow', 'red', 'purple']
    type = ['conv_cite', 'conv', 'resnet', 'unet', 'unet2']
    for i in range(5):
        data = []
        for j in range(5):
            data.append(aucs[j][i])
        mean, var = mean_variance(data)
        plt.plot(x_data, data, color=colors[i], label=type[i]+" ("+str(mean)+")")
    plt.legend()
    plt.savefig(os.path.join(pic_path, png_name))
    plt.show()


def draw_aucs():
    # AD vs NC
    t1_auc = [0.99, 0.98, 0.99, 0.93, 0.91]
    t2_auc = [0.96, 0.92, 0.89, 0.81, 0.95]
    t3_auc = [0.95, 0.92, 0.91, 0.98, 0.89]
    t4_auc = [0.88, 0.88, 0.79, 0.88, 0.87]
    t5_auc = [0.76, 0.85, 0.76, 0.87, 0.89]
    aucs = [t1_auc, t2_auc, t3_auc, t4_auc, t5_auc]
    draw_multi_auc(aucs, 'AD vs NC', 'NCvsADAUC.png')

    # NC vs EM
    t1_auc = [0.93, 1.0, 0.94, 0.95, 0.94]
    t2_auc = [0.93, 0.68, 0.79, 0.76, 0.63]
    t3_auc = [0.99, 0.88, 0.96, 0.93, 0.77]
    t4_auc = [0.97, 0.78, 0.97, 0.96, 0.90]
    t5_auc = [0.6, 0.49, 0.5, 0.5, 0.55]
    aucs = [t1_auc, t2_auc, t3_auc, t4_auc, t5_auc]
    draw_multi_auc(aucs, 'NC vs EM', 'NCvsEMAUC.png')
    #
    # AD vs LM
    t1_auc = [0.94, 0.94, 0.89, 0.76, 0.88]
    t2_auc = [0.81, 0.65, 0.81, 0.84, 0.66]
    t3_auc = [0.98, 0.89, 0.95, 0.93, 0.79]
    t4_auc = [0.83, 0.72, 0.79, 0.65, 0.65]
    t5_auc = [0.92, 0.87, 0.86, 0.77, 0.89]
    aucs = [t1_auc, t2_auc, t3_auc, t4_auc, t5_auc]
    draw_multi_auc(aucs, 'AD vs LM', 'AD vs LMAUC.png')

    # EM vs LM
    t1_auc = [0.92, 0.92, 0.88, 0.97, 0.96]
    t2_auc = [0.89, 0.95, 0.98, 0.99, 0.96]
    t3_auc = [0.984, 0.81, 0.94, 0.92, 0.87]
    t4_auc = [0.9, 0.79, 0.56, 0.7, 0.67]
    t5_auc = [0.97, 0.92, 0.97, 0.92, 0.93]
    aucs = [t1_auc, t2_auc, t3_auc, t4_auc, t5_auc]
    draw_multi_auc(aucs, 'EM vs LM', 'EMvsLMAUC.png')


def cal_all_info():
    path1 = "/home/fan/Desktop/ad_classify/log/EMvsLM/T5/conv/result.md"  # conv_cite
    path2 = "/home/fan/Desktop/ad_classify/log/EMvsLM/T5/conv1/result.md"
    path3 = "/home/fan/Desktop/ad_classify/log/EMvsLM/T5/resnet/result.md"
    path4 = "/home/fan/Desktop/ad_classify/log/EMvsLM/T5/unet1/result.md"
    path5 = "/home/fan/Desktop/ad_classify/log/EMvsLM/T5/unet/result.md"  # u2

    paths = []
    paths.append(path1)
    paths.append(path2)
    paths.append(path3)
    paths.append(path4)
    paths.append(path5)

    models = ['conv_cite', 'conv', 'resnet', 'unet', 'unet2']
    i = 0
    for path in paths:
        print('\n', models[i])
        print("sen ", two_sen(path, 0.5))
        print("spe ", two_spe(path, 0.5))
        print("ppv ", two_ppv(path))
        print("npv ", two_npv(path))
        sens, minus_spes = two_roc(path)
        print('auc ', auc(sens))
        i += 1


def four_sen(path, index=0):
    true_positive = 0
    positive = 0

    data = read_data(path)
    for i in range(0, len(data), 2):
        tmp = data[i]
        tmp = tmp.split(" ")
        tmp[0] = tmp[0].replace("[", "")
        tmp[-1] = tmp[-1].replace("]", "")
        tmp[-1] = tmp[-1].strip()

        tmp1 = data[i+1]
        tmp1.replace("  ", " ")
        tmp1 = tmp1.split(" ")
        tmp1[0] = tmp1[0].replace("[", "")
        tmp1[-1] = tmp1[-1].replace("]", "")
        tmp1[-1] = tmp1[-1].strip()

        tmp = float(round(float(tmp[0]), 5))
        tmp2 = []
        for t in tmp1:
            try:
                tmp2.append(float(round(float(t), 5)))
            except:
                continue

        if abs(tmp - tmp2[index]) < 0.00001:
            positive+=1
        else:
            continue

        flag = True
        for tmp in tmp2:
            if tmp - tmp2[index] > 0.00001:
                flag = False
        if flag:
            true_positive += 1
    return round(true_positive / positive, 2)


def four_spe(path, index=0):
    non_positive = 0  # number in original data
    true_non_positive = 0  # predict and it's true
    data = read_data(path)

    for i in range(0, len(data), 2):
        tmp = data[i]
        tmp = tmp.split(" ")
        tmp[0] = tmp[0].replace("[", "")
        tmp[-1] = tmp[-1].replace("]", "")
        tmp[-1] = tmp[-1].strip()

        tmp1 = data[i+1]
        tmp1.replace("  ", " ")
        tmp1 = tmp1.split(" ")
        tmp1[0] = tmp1[0].replace("[", "")
        tmp1[-1] = tmp1[-1].replace("]", "")
        tmp1[-1] = tmp1[-1].strip()

        tmp = float(round(float(tmp[0]), 5))
        tmp2 = []
        for t in tmp1:
            try:
                tmp2.append(float(round(float(t), 5)))
            except:
                continue

        if abs(tmp - tmp2[index]) > 0.00001:
            non_positive += 1
        else:
            continue

        flag = False
        for t in tmp2:
            if t - tmp2[index] > 0.00001:
                flag = True
        if flag:
            true_non_positive += 1
    # print(true_non_positive, " ", non_positive)
    return round(true_non_positive / non_positive, 2)


def four_acc(path, index=0):
    total = 0
    data = read_data(path)
    acc = 0

    for i in range(0, len(data), 2):
        tmp = data[i]
        tmp = tmp.split(" ")
        tmp[0] = tmp[0].replace("[", "")
        tmp[-1] = tmp[-1].replace("]", "")
        tmp[-1] = tmp[-1].strip()

        tmp1 = data[i + 1]
        tmp1.replace("  ", " ")
        tmp1 = tmp1.split(" ")
        tmp1[0] = tmp1[0].replace("[", "")
        tmp1[-1] = tmp1[-1].replace("]", "")
        tmp1[-1] = tmp1[-1].strip()

        tmp = float(round(float(tmp[0]), 5))
        tmp2 = []
        for t in tmp1:
            try:
                tmp2.append(float(round(float(t), 5)))
            except:
                continue

        if abs(tmp - tmp2[index]) < 0.00001:
            total += 1
        else:
            continue
        flag = True
        for t in tmp2:
            if t - tmp2[index] > 0.00001:
                flag = False
        if flag:
            acc += 1
    # print(acc, " ", total)
    return round(acc / total, 2)


def four_spe_sen():
    sen_ad_conv_cite = [0.82, 0.86, 0.91, 1.0, 0.73]
    spe_ad_conv_cite = [0.98, 0.97, 1.0, 1.0, 0.97]

    # print(mean_variance(sen_ad_conv_cite))
    # print(mean_variance(spe_ad_conv_cite))

    sen_ad_conv = [0.91, 0.82, 0.64, 0.79, 0.68]
    spe_ad_conv = [0.97, 0.93, 0.96, 0.9, 0.99]

    # print(mean_variance(sen_ad_conv))
    # print(mean_variance(spe_ad_conv))

    sen_ad_res = [0.91, 0.92, 0.86, 1.0, 0.73]
    spe_ad_res = [0.65, 0.91, 0.99, 1.0, 0.9]

    # print(mean_variance(sen_ad_res))
    # print(mean_variance(spe_ad_res))

    sen_ad_unet = [0.82, 0.82, 0.82, 0.83, 0.73]
    spe_ad_unet = [0.95, 0.84, 0.96, 1.0, 0.93]

    # print(mean_variance(sen_ad_unet))
    # print(mean_variance(spe_ad_unet))

    sen_ad_unet2 = [1.0, 0.86, 0.91, 1.0, 0.95]
    spe_ad_unet2 = [0.92, 0.91, 0.97, 1.0, 0.91]

    # print(mean_variance(sen_ad_unet2))
    # print(mean_variance(spe_ad_unet2))

    sen_cn_conv_cite = [0.95, 0.92, 0.92, 1.0, 1.0]
    spe_cn_conv_cite = [0.91, 0.97, 0.96, 1.0, 0.88]

    # print(mean_variance(sen_cn_conv_cite))
    # print(mean_variance(spe_cn_conv_cite))

    sen_cn_conv = [0.86, 0.62, 0.79, 0.79, 0.88]
    spe_cn_conv = [0.92, 0.88, 0.85, 0.96, 0.82]

    # print(mean_variance(sen_cn_conv))
    # print(mean_variance(spe_cn_conv))

    sen_cn_res = [0.41, 0.67, 0.78, 1.0, 0.96]
    spe_cn_res = [1.0, 1.0, 0.93, 1.0, 0.8]

    # print(mean_variance(sen_cn_res))
    # print(mean_variance(spe_cn_res))

    sen_cn_unet = [0.82, 0.71, 0.62, 0.88, 0.96]
    spe_cn_unet = [0.94, 0.94, 0.97, 0.9, 0.8]

    # print(mean_variance(sen_cn_unet))
    # print(mean_variance(spe_cn_unet))

    sen_cn_unet2 = [0.86, 0.83, 0.92, 1.0, 0.71]
    spe_cn_unet2 = [1.0, 0.91, 0.92, 1.0, 0.95]

    # print(mean_variance(sen_cn_unet2))
    # print(mean_variance(spe_cn_unet2))

    sen_em_conv_cite = [0.91, 1.0, 0.91, 1.0, 0.9]
    spe_em_conv_cite = [1.0, 0.99, 0.96, 1.0, 0.93]

    # print(mean_variance(sen_em_conv_cite))
    # print(mean_variance(spe_em_conv_cite))

    sen_em_conv = [0.76, 0.75, 0.82, 0.91, 0.75]
    spe_em_conv = [0.97, 0.99, 0.84, 0.99, 0.89]

    # print(mean_variance(sen_em_conv))
    # print(mean_variance(spe_em_conv))

    sen_en_res = [0.41, 1.0, 0.86, 1.0, 0.5]
    spe_em_res = [1.0, 0.91, 0.91, 1.0, 0.8]

    # print(mean_variance(sen_en_res))
    # print(mean_variance(spe_em_res))

    sen_em_unet = [0.86, 1.0, 0.91, 0.91, 0.6]
    spe_em_unet = [0.95, 0.94, 0.81, 1.0, 0.9]

    # print(mean_variance(sen_em_unet))
    # print(mean_variance(spe_em_unet))

    sen_em_unet2 = [0.86, 0.85, 0.82, 1.0, 0.75]
    spe_em_unet2 = [1.0, 0.99, 0.96, 1.0, 0.87]

    # print(mean_variance(sen_em_unet2))
    # print(mean_variance(spe_em_unet2))

    sen_lm_conv_cite = [1.0, 1.0, 1.0, 1.0, 0.67]
    spe_lm_conv_cite = [1.0, 1.0, 1.0, 1.0, 0.98]

    # print(mean_variance(sen_lm_conv_cite))
    # print(mean_variance(spe_lm_conv_cite))

    sen_lm_conv = [0.95, 0.88, 0.62, 0.92, 0.67]
    spe_lm_conv = [0.95, 0.89, 0.97, 0.96, 0.97]

    # print(mean_variance(sen_lm_conv))
    # print(mean_variance(spe_lm_conv))

    sen_lm_res = [0.73, 0.92, 0.88, 1.0, 0.54]
    spe_lm_res = [0.83, 0.97, 0.96, 1.0, 0.94]

    # print(mean_variance(sen_lm_res))
    # print(mean_variance(spe_lm_res))

    sen_lm_unet = [0.82, 0.62, 0.88, 0.88, 0.54]
    spe_lm_unet = [0.92, 0.98, 1.0, 0.93, 0.98]

    # print(mean_variance(sen_lm_unet))
    # print(mean_variance(spe_lm_unet))

    sen_lm_unet2 = [0.95, 0.79, 0.79, 1.0, 0.67]
    spe_lm_unet2 = [0.97, 0.97, 0.79, 1.0, 0.95]

    print(mean_variance(sen_lm_unet2))
    print(mean_variance(spe_lm_unet2))


def two_acc_com():
    conv_cite = [89.58, 94.15, 89.54, 93.04]
    conv = [80.47, 85.2, 79.73, 92.68]
    resnet = [82.69, 89.23, 80.38, 90.98]
    unet = [86.16, 93.26, 84.79, 93.11]
    unet2 = [83.03, 88.75, 85.76, 91.27]
    arr = [conv_cite, conv, resnet, unet, unet2]
    x_data = ['NC vs EM', 'EM vs LM', 'LM vs AD', 'AD vs NC']
    name = ['conv_cite', 'conv', 'resnet', 'unet', 'unet2']
    colors = ['orange', 'green', 'blue', 'red', 'yellow']
    plt.figure()
    plt.title('accuracy compare')
    plt.xlabel("different task")
    plt.ylabel("accuracy rate (%)")
    for i in range(5):
        mean, var = mean_variance(arr[i])
        plt.plot(x_data, arr[i], color=colors[i], label=name[i]+' ('+str(mean)+", "+str(var)+')')
    plt.legend()
    plt.savefig(os.path.join(pic_path, "AccCompare.png"))
    plt.show()


def auc_com():
    conv_cite = [0.88, 0.93, 0.90, 0.91]
    conv = [0.77, 0.88, 0.81, 0.91]
    resnet = [0.83, 0.87, 0.86, 0.87]
    unet = [0.82, 0.90, 0.79, 0.89]
    unet2 = [0.76, 0.88, 0.77, 0.90]
    arr = [conv_cite, conv, resnet, unet, unet2]
    x_data = ['NC vs EM', 'EM vs LM', 'LM vs AD', 'AD vs NC']
    name = ['conv_cite', 'conv', 'resnet', 'unet', 'unet2']
    colors = ['orange', 'green', 'blue', 'red', 'yellow']
    plt.figure()
    plt.title('auccompare')
    plt.xlabel("different task")
    plt.ylabel("AUC")
    for i in range(5):
        mean, var = mean_variance(arr[i])
        plt.plot(x_data, arr[i], color=colors[i], label=name[i] + ' (' + str(mean) + ')')
    plt.legend()
    plt.savefig(os.path.join(pic_path, "AUCCompare.png"))
    plt.show()


def draw_bar():

    # conv = [90.91, 94.12, 87.14, 89.71, 78.57]
    # resnet = [80.3, 85.29, 90.0, 88.24, 61.14]
    # unet = [89.39, 88.24, 88.57, 89.71, 71.43]
    # unet2 = [95.45, 92.65, 88.57, 83.82, 62.86]

    conv = [87.5, 76.67, 71.74, 85.11, 74.44]
    resnet = [61.36, 84.44, 84.78, 100, 64.44]
    unet = [82.95, 77.78, 80.43, 87.23, 76.67]
    unet2 = [92.05, 83.33, 85.87, 100, 71.11]

    name_list = [1, 2, 3, 4, 5]
    x = list(range(len(unet)))
    total_width, n = 0.8, 4
    width = total_width / n
    plt.title("NC vs EM vs LM vs AD")
    plt.ylabel("accuracy rate (%)")
    plt.xlabel("number of corss validation")
    mean, var = mean_variance(conv)
    plt.bar(x, conv, width=width, label='convnet'+' ('+str(mean)+", "+str(var)+')', color='green')
    for i in range(len(x)):
        x[i] += width
    mean, var = mean_variance(resnet)
    plt.bar(x, resnet, width=width, label='resnet'+' ('+str(mean)+", "+str(var)+')', color='blue')
    for i in range(len(x)):
        x[i] += width
    mean, var = mean_variance(unet)
    plt.bar(x, unet, width=width, label='unet'+' ('+str(mean)+", "+str(var)+')', color='red')
    for i in range(len(x)):
        x[i] += width
    mean, var = mean_variance(unet2)
    plt.bar(x, unet2, width=width, label='unet2'+' ('+str(mean)+", "+str(var)+')', color='yellow', tick_label=name_list)
    plt.legend()
    plt.savefig(os.path.join(pic_path, "four_bar.png"))
    plt.show()


if __name__ == '__main__':
    pic_path = ""
    path1 = r"F:\ZDQMRI\谷歌ADNI1NC"
    path2 = r"F:\reADNC\NC"