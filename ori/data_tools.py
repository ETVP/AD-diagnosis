import os
import numpy as np
import shutil
import nibabel as nib
import re
import random


def rename(files, number):
    for file in files:
        tmp = file
        file = file.split('.')
        os.rename(os.path.join(path, tmp), os.path.join(path, str(file[0]) + '.' + str(file[1]) + '.5.npy'))


def augment_contrast(data_sample, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True):
    if not per_channel:
        mn = data_sample.mean()
        if preserve_range:
            minm = data_sample.min()
            maxm = data_sample.max()
        if np.random.random() < 0.5 and contrast_range[0] < 1:
            factor = np.random.uniform(contrast_range[0], 1)
        else:
            factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
        data_sample = (data_sample - mn) * factor + mn
        if preserve_range:
            data_sample[data_sample < minm] = minm
            data_sample[data_sample > maxm] = maxm
    else:
        for c in range(data_sample.shape[0]):
            mn = data_sample[c].mean()
            if preserve_range:
                minm = data_sample[c].min()
                maxm = data_sample[c].max()
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
            data_sample[c] = (data_sample[c] - mn) * factor + mn
            if preserve_range:
                data_sample[c][data_sample[c] < minm] = minm
                data_sample[c][data_sample[c] > maxm] = maxm
    return data_sample


def exchange(file1, file2, number):
    # move one test data
    files = os.listdir(file1)
    for file in files:
        tmp = file.split('.')
        if tmp[2] == str(number):
            shutil.move(os.path.join(file1, file), os.path.join(file2, file))


def view_data_dim(path):
    files = os.listdir(path)
    for file in files:
        data = nib.load(os.path.join(path, file)).get_data()
        print(data.shape)


def read_not_zero_dim(path):
    """
    170 * 256 * 256
    :param path:
    :return:
    """
    files = os.listdir(path)
    min_x = 170
    min_y = min_z = 256
    for file in files:
        if not file.endswith(".nii"):
            continue
        img = nib.load(os.path.join(path, file)).get_data()
        img = np.squeeze(img)
        count = [0, 0, 0]
        print(np.min(img))
        for i in range(len(img)):
            if img[i].any() <= np.median(img[i]):
                count[0] += 1
        for i in range(len(img[0])):
            if img[:, i, :].any() <= np.median(img[:,i,:]):
                count[1] += 1
        for i in range(len(img[0][0])):
            if img[:, :, i].any() <= np.median(img[:,:,i]):
                count[2] += 1
        if min_x > count[0]:
            min_x = count[0]
        if min_y > count[1]:
            min_y = count[1]
        if min_z > count[2]:
            min_z = count[2]
        print(count)
    # ADNC20200225 train: 51 82 116
    # ADNC20200225 val 52 86 124
    # ADNC20200225 test 52 86 121
    # we can save the npy dim should be 120, 168, 136
    # F:\ADNIADMPRAGESIEMENS\CN\reMNI\skull 24 53 1
    # F:\ADNIADMPRAGESIEMENS\AD\reMNI\skull 24 55 1
    # F:\ADNIADMPRAGESIEMENS\ -> 172 220 156 -> 144, 168, 152
    print(min_x, " ", min_y, " ", min_z)

def read_nii_info(path):
    img = nib.load(path)
    print(img)


def read_md(path):
    file = open(path)
    arr = []
    while True:
        try:
            line = file.readline()
            if line.__contains__("end"):
                file.close()
                break
            line = re.split('[ , \\n]', line)
            for item in line:
                if item == '':
                    continue
                try:
                    arr.append(float(item))
                except:
                    continue
        except:
            file.close()
    print("num: ", len(arr))
    return arr


def rename_adni1_nii_file(path, type):
    p = re.compile(r'\d{17}')
    p1 = re.compile(r'\d{3}_[VS]_\d{4}')
    i = 1
    for file in os.listdir(path):
        date = re.findall(p, file)
        subject = re.findall(p1, file)
        # print(os.path.join(path, type+"_"+subject[0]+"_"+date[0][:8]+"_"+str(i)+".nii"))
        os.rename(os.path.join(path, file), os.path.join(path, type+"_"+subject[0]+"_"+date[0][:8]+"_"+str(i)+".nii"))
        i += 1


def split_subject(data_path):
    """
    split dataset through the subjects num
    :param data_path: list, 0: original data path, 1: train file path 2: val file path 3: test file path
    :return:
    """
    files = []
    for f in os.listdir(data_path[0]):
        if not os.path.isdir(os.path.join(data_path[0], f)):
            files.append(f)
    p = re.compile(r'\d{3}_{0,1}[SV]_{0,1}\d{4}')
    AD_subjects = set()
    CN_subjects = set()
    for f in files:
        if not f.endswith("nii"):
            continue
        if f.startswith("AD"):
            AD_subjects.add(re.findall(p, f)[0])
        elif f.startswith("CN"):
            CN_subjects.add(re.findall(p, f)[0])
    AD_subjects = list(AD_subjects)
    CN_subjects = list(CN_subjects)
    random.shuffle(AD_subjects)
    random.shuffle(CN_subjects)
    for f in os.listdir(data_path[0]):
        if not os.path.isdir(os.path.join(data_path[0], f)):
            sub = re.findall(p, f)[0]
            if f.startswith("AD"):
                if sub in AD_subjects[: int(len(AD_subjects) / 10) * 7]:
                    shutil.move(os.path.join(data_path[0], f), os.path.join(data_path[1], f))
                elif sub in AD_subjects[int(len(AD_subjects) / 10) * 7: int(len(AD_subjects) / 10) * 8]:
                    shutil.move(os.path.join(data_path[0], f), os.path.join(data_path[2], f))
                else:
                    shutil.move(os.path.join(data_path[0], f), os.path.join(data_path[3], f))
            elif f.startswith("CN"):
                if sub in CN_subjects[: int(len(CN_subjects) / 10) * 7]:
                    shutil.move(os.path.join(data_path[0], f), os.path.join(data_path[1], f))
                elif sub in CN_subjects[int(len(CN_subjects) / 10) * 7: int(len(CN_subjects) / 10) * 8]:
                    shutil.move(os.path.join(data_path[0], f), os.path.join(data_path[2], f))
                else:
                    shutil.move(os.path.join(data_path[0], f), os.path.join(data_path[3], f))


def data_augment(path):
    """
    use data augment to balance the number of AD and CN
    :param path:
    :return:
    """
    train_ad_file = []
    train_cn_file = []
    val_ad_file = []
    val_cn_file = []
    paths = [path+"/train", path+"/val"]
    for f in os.listdir(paths[0]):
        if f.startswith("AD"):
            train_ad_file.append(f)
        elif f.startswith("CN"):
            train_cn_file.append(f)
    for f in os.listdir(paths[1]):
        if f.startswith("AD"):
            val_ad_file.append(f)
        elif f.startswith("CN"):
            val_cn_file.append(f)


def nii2npy(data_path):
    for f in os.listdir(data_path):
        data = nib.load(os.path.join(data_path, f)).get_data()
        data = np.squeeze(data)
        np.save(os.path.join(data_path, f.replace(".nii", ".npy")), data)


def mins_dim(data_path):
    for f in os.listdir(data_path):
        data = np.load(os.path.join(data_path, f))
        data = np.squeeze(np.asarray(data))
        data = data[1:-1, :, :]
        print(data.shape)
        np.save(os.path.join(data_path, f), data)


if __name__ == '__main__':
    path = r"F:\ADNIADMPRAGESIEMENS\AD\reMNI\skull"
    read_not_zero_dim(path)
