from keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import nibabel as nib


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
img_file = r"./"
model = ""
# model = load_model("log/unet.loss.b5.h5")
# model.summary()
predict_result = []
sigmoid = True
num = 0
files = os.listdir(img_file)
true_path = "./logs/true"
true_file = open(true_path, "a+")


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
        img_data = img_data[np.newaxis, :, :, :, np.newaxis]
        predict = model.predict(img_data)
        print(predict, " ", labels[i])
        if predict[num][0] <= 0.5 and labels[i] == 0:
            true_num += 1
            print(os.path.join(img_file, file), file=true_file)
        elif predict[num][0] > 0.5 and labels[i] == 1:
            true_num += 1
            print(os.path.join(img_file, file), file=true_file)
        else:
            false_num += 1
        i += 1

    print("True: ", true_num)
    print("False: ", false_num)
    print(true_num/(false_num + true_num))


def zero_replace(data, slice):
    """
    Replace all specified slices with 0
    :param data:
    :param slice: a array
    :return:
    """
    if type(slice) == int:
        data[:, :, slice] = 0.1
        return data
    for i in slice:
        data[:, :, i] = 0.1
    return data


def x_zero_replace(data, x):
    """

    :return:
    """
    if type(x) == int:
        data[x, :, :] = 0.1
        return data
    for i in x:
        data[i, :, :] = 0.1
    return data


def y_zero_replace(data, y):
    """

    :return:
    """
    if type(y) == int:
        data[:, y, :] = 0.1
        return data
    for i in y:
        data[:, i, :] = 0.1
    return data


def gauss_blur(data, slice):
    """

    :param data:
    :param slice:
    :return:
    """
    if type(slice) == int:
        data[:, :, slice] = generate_gauss_noise(data[:, :, slice], 25)
        data[:, :, slice] = (data[:, :, slice] - np.min(data[:, :, slice])) / \
                            (np.max(data[:, :, slice]) - np.min(data[:, :, slice]))
        return data
    for i in slice:
        data[:, :, i] = generate_gauss_noise(data[:, :, i], 25)
        data[:, :, i] = (data[:, :, i] - np.min(data[:, :, i])) / \
                            (np.max(data[:, :, i]) - np.min(data[:, :, i]))
    return data


def generate_gauss_noise(data, noise_sigma):
    h = data.shape[0]
    w = data.shape[1]
    noise = np.random.randn(h, w) * noise_sigma
    return noise


def read_file(path):
    files = []
    f = open(path, "r")
    for line in f.readlines():
        if line.__contains__(".npy"):
            files.append(line.replace("\n", ""))
    return files


def test(path):
    for f in os.listdir(path):
        print(os.path.join(path, f), file=true_file)


def evaluate_slice(model, files):
    """
    评估每张切片的影响
    :param model: 加载好的模型
    :param files: 存放的测试文件
    :return:
    """
    for file in files:
        img_data = np.load(file)
        img_data = img_data[np.newaxis, :, :, :, np.newaxis]
        predict = model.predict(img_data)
        print(file, " \n")
        for i in range(img_data.shape[2]):
            tmp_img_data = zero_replace(np.squeeze(img_data), i)
            tmp_img_data = tmp_img_data[np.newaxis, :, :, :, np.newaxis]
            a_predict = model.predict(tmp_img_data)
            print("slice: ", i, "predict: ", predict, " zero_predict: ", a_predict, " effect: ",
                  abs(predict-a_predict))


def evaluate_axis(model, files, axis):
    """
    评估沿x轴的影响
    :param model:
    :param files:
    :return:
    """
    for file in files:
        img_data = np.load(file)
        img_data = img_data[np.newaxis, :, :, :, np.newaxis]
        predict = model.predict(img_data)
        print(file, " \n")
        if axis == 0:
            for i in range(img_data.shape[1]):
                tmp_img_data = x_zero_replace(np.squeeze(img_data), i)
                tmp_img_data = tmp_img_data[np.newaxis, :, :, :, np.newaxis]
                a_predict = model.predict(tmp_img_data)
                print("slice: ", i, "predict: ", predict, " x_zero_predict: ", a_predict, " effect: ",
                      abs(predict - a_predict))
        else:
            for i in range(img_data.shape[2]):
                tmp_img_data = y_zero_replace(np.squeeze(img_data), i)
                tmp_img_data = tmp_img_data[np.newaxis, :, :, :, np.newaxis]
                a_predict = model.predict(tmp_img_data)
                print("slice: ", i, "predict: ", predict, " x_zero_predict: ", a_predict, " effect: ",
                      abs(predict - a_predict))



def evaluate_slice_effect(file):
    """
    首先将evaluate_slice的输出存为文件，然后使用该函数来通过文件来获得每个图像每张切片的影响
    :param file:
    :return:
    """
    f = open(file, "r")
    files = []
    file_num = 0
    slice_effects = []
    slice_effect = []
    for line in f.readlines():
        if line.__contains__(".npy"):
            files.append(line.strip())
            if not file_num == 0:
                slice_effects.append(slice_effect)
            slice_effect = []
            file_num += 1
        if line.startswith("slice:"):
            items = line.split("[[")
            slice_effect.append(float(items[-1].split("]]")[0]))
    slice_effects.append(slice_effect)
    f.close()
    return files, slice_effects


def sort_slice_by_slice_effect(data):
    """
    根据切片影响的大小来进行切片排序
    :param data: one dim array
    :return:
    """
    sorted_data = sorted(data)
    slices = []
    for item in sorted_data:
        for i in range(len(data)):
            if data[i] == item and i not in slices:
                slices.append(i)
                break
    return slices


def get_data_center(path, need_x, need_y, need_z):
    img = nib.load(path).get_data()
    img = np.squeeze(img)
    min_ = [0, 0, 0]
    max_ = [img.shape[0], img.shape[1], img.shape[2]]
    for i in range(len(img)):
        if img[i].any() == 0:
            min_[0] = i
        else:
            break
    for i in range(len(img)-1, -1, -1):
        if img[i].any() == 0:
            max_[0] = i
        else:
            break
    for i in range(len(img[0])):
        if img[:,i,:].any() == 0:
            min_[1] = i
        else:
            break
    for i in range(len(img[0])-1, -1, -1):
        if img[:,i,:].any() == 0:
            max_[1] = i
        else:
            break
    for i in range(len(img[0][0])):
        if img[:,:,i].any() == 0:
            min_[2] = i
        else:
            break
    for i in range(len(img[0][0])-1,-1,-1):
        if img[:,:,i].any() == 0:
            max_[2] = i
        else:
            break
    center = [int((min_[0]+max_[0])/2), int((min_[1]+max_[1])/2), int((min_[2]+max_[2])/2)]
    if center[0] - need_x < 0:
        center[0] = need_x
    elif center[0] + need_x > img.shape[0]:
        center[0] = img.shape[0]-need_x
    if center[1] - need_y < 0:
        center[1] = need_y
    elif center[1] + need_y > img.shape[1]:
        center[1] = img.shape[1] - need_y
    if center[2] - need_z < 0:
        center[2] = need_z
    elif center[2] + need_z > img.shape[2]:
        center[2] = img.shape[2] - need_z
    return center


def get_real_slice(ori_file, files, slices):
    for i in range(len(files)):
        f = files[i].split("/")[-1]
        real_file_path = os.path.join(ori_file, f.replace(".npy", ".nii"))
        center = get_data_center(real_file_path, 72, 84, 76)
        slices[i, :] = slices[i, :] + center[-1]-76
        return slices


def stat_freq(file):
    """
    statistics the frequency of the 40 most affected slices
    :return:
    """
    f = open(file, "r")
    stat = [0]*152
    for line in f.readlines():
        if line.startswith("["):
            data = line.strip().replace("[", "").replace("]", "").replace(" ", "")
            data = data.split(",")
            for i in range(len(data)-50, len(data)):
                stat[int(data[i])] += 1
    stat_sorted = sorted(stat)
    slices = []
    for i in range(len(stat_sorted)-40, len(stat_sorted)):
        for j in range(len(stat)):
            if stat[j] == stat_sorted[i] and j not in slices:
                slices.append(j)
                break
    print(slices)


def stat_slice(slices):
    sl = [0 for i in range(152)]
    for ss in slices[-50:]:
        for item in ss:
            sl[item] += 1
    print(sl)


def evaluateSliceProcess():
    two_classify_predict()
    f = open(true_path, 'r')
    files = []
    for file in f.readlines():
        files.append(file.replace("\n", ""))
    evaluate_slice(model, files)
    files, slice_effects = evaluate_slice_effect(file)
    sorted_slice = []
    for i in range(len(slice_effects)):
        sorted_slice.append(sort_slice_by_slice_effect(slice_effects[i]))
    # get_real_slice(ori_file, files, slices)
    # stat_freq(file)

if __name__ == '__main__':
    # 选取80-125的切片
    # two_classify_predict()
    true_file.close()

