import os
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import re


def move_file(path1, path2):
    for top, dirs, nondirs in os.walk(path1):
        for d in nondirs:
            print(os.path.join(top, d))
            open(os.path.join(path2, d), 'wb').write(open(os.path.join(top, d), 'rb').read())


def rename(path: str, type):
    for top, dirs, nondirs in os.walk(path):
        for d in nondirs:
            target_d = d.replace("ADNI", type)
            os.rename(os.path.join(top, d), os.path.join(top, target_d))


def plot_slice_stat(effect):
    """
    绘制切片的影响
    :param effect:
    :return:
    """
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2. - 0.2, 1.03 * height, '%s' % "")
    autolabel(plt.bar(range(len(effect)), effect, color='rgb'))
    plt.savefig('effect.png')


def plot_stat(name_list, effect):
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2. - 0.2, 1.03 * height, '%s' % int(height))
    autolabel(plt.bar(range(len(effect)), effect, color='rgb', tick_label=name_list))
    plt.show()


def plot(path):
    img = nib.load(path).get_data()
    img = img[21:71, 50:81, 94:144]
    plt.figure()
    for i in range(50):
        plt.imshow(img[i, :, :], cmap='gray')
        plt.show()


def mark_subdataset(path, num):
    for f in os.listdir(path):
        os.rename(os.path.join(path, f), os.path.join(path, f.replace(".npy", "."+str(num)+".npy")))


def get_excel_data(path):
    """
    根据AIBLexcel来提取AD CN subject
    :param path:
    :return:
    """
    df = pd.read_excel(path)
    extracted_data = df.ix[:, ["RID", "MMSCORE"]]
    data = []
    for i in range(len(extracted_data)):
        data.append((extracted_data.ix[i][0], extracted_data.ix[i][1]))
    return data


def del_old_cn(path):
    numbers = []
    for file in os.listdir(path):
        if file.startswith("cn"):
            numbers.append(int(file.split(".")[1]))
    numbers.sort()
    numbers = numbers[1::2]
    files = os.listdir(path)
    for f in files:
        if str(f.split(".")[1]) in numbers:
            os.remove(os.path.join(path, f))


if __name__ == '__main__':
    # path1 = r"F:\AIBL\AIBL"
    # path = r"F:\AIBL\AD"
    # move_file(path1, path)
    excel_path = r"D:\AIBL\Data_extract_3.3.0\aibl_mmse_01-Jun-2018.xlsx"
    data_path = r"D:\AIBL\AIBL"
    data = get_excel_data(excel_path)
    AD = []
    CN = []
    for item in data:
        if item[1] >= 27:
            CN.append(item[0])
        elif item[1] <= 9:
            AD.append(item[0])
    print(AD)