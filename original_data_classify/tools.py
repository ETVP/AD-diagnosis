import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import shutil
import re
import random


def get_data(path, need_x, need_y, need_z):
    img = nib.load(path).get_data()
    img = np.squeeze(img)
    center = get_center(path, need_x, need_y, need_z)
    img = img[center[0]-need_x:center[0]+need_x, center[1]-need_y:center[1]+need_y,
           center[2]-need_z:center[2]+need_z]
    _max = np.max(img)
    _min = np.min(img)
    img = (img - _min) / (_max - _min)
    return img


def get_center(path, need_x, need_y, need_z):
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


def get_data(path, begin_slice, end_slice):
    img = nib.load(path).get_data()
    img = np.squeeze(img)
    return img[begin_slice: end_slice+1]


def data_flip(data):
    """
    all axies
    :param data:
    :return:
    """
    return np.flip(data)


def data_rotate(data, angle):
    if angle == 90:
        return np.rot90(data, -1)
    elif angle == 180:
        return np.rot90(data, 2)
    elif angle == 270:
        return np.rot90(data, 1)


def gauss_flur(data, sigma):
    """
    week gauss flur
    :param data:
    :return:
    """
    for i in range(len(data[0][0])):
        data[:,:,i] = filters.gaussian_filter(data[:,:,i], sigma)
    return data


def draw_data(data):
    plt.figure()
    plt.subplot(131)
    plt.imshow(data[60, :, :], cmap="gray")
    plt.subplot(132)
    plt.imshow(data[:, 84, :], cmap="gray")
    plt.subplot(133)
    plt.imshow(data[:, :, 68], cmap="gray")
    plt.show()


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


def find_files(original_path, result_path):
    """
    find the files of subdataset, and write them in the result_file
    :param original_path:
    :param result_path:
    :return:
    """
    result_train_file = os.path.join(result_path, "train_files")
    result_val_file = os.path.join(result_path, "val_files")
    result_test_file = os.path.join(result_path, "test_files")
    f = open(result_train_file, "w")
    for item in os.listdir(os.path.join(original_path, "train")):
        print(item, file=f)
    f.close()
    f = open(result_val_file, "w")
    for item in os.listdir(os.path.join(original_path, "val")):
        print(item, file=f)
    f.close()
    f = open(result_test_file, "w")
    for item in os.listdir(os.path.join(original_path, "test")):
        print(item, file=f)
    f.close()


def move_item2subdataset(files, data_path, result_path):
    """
    move nii files to the subdataset according to the files
    :param files: a array consist of train_file, val_file and test_file
    :param data_path:
    :param result_path:
    :return:
    """
    f = open(files[0], "r")
    data = f.readlines()
    for item in data:
        item = item.strip()
        item = item.replace(".npy", ".nii")
        shutil.move(os.path.join(data_path, item), os.path.join(result_path, "train", item))
    f.close()
    f = open(files[1], "r")
    data = f.readlines()
    for item in data:
        item = item.strip()
        item = item.replace(".npy", ".nii")
        shutil.move(os.path.join(data_path, item), os.path.join(result_path, "val", item))
    f.close()
    f = open(files[2], "r")
    data = f.readlines()
    for item in data:
        item = item.strip()
        item = item.replace(".npy", ".nii")
        shutil.move(os.path.join(data_path, item), os.path.join(result_path, "test", item))
    f.close()


def remove_repeat_subject(path):
    files = []
    for file in os.listdir(path):
        files.append(file)
    files.sort()
    p = re.compile(r'\d{3}_{0,1}[VS]_{0,1}\d{4}')
    subjects = []
    for f in files:
        if not f.endswith(".nii"):
            continue
        subject = re.findall(p, f)[0]
        if subject in subjects:
            shutil.move(os.path.join(path, f), os.path.join(path, "tmp", f))
        subjects.append(subject)


def random_del(path, num, prefix, suffix):
    """
    for balance dataset, we can delete some augmented data
    :param path:
    :param num:
    :param prefix:
    :param suffix:
    :return:
    """
    files = []
    for f in os.listdir(path):
        if f.startswith(prefix):
            flag = True
            for s in suffix:
                if f.__contains__(s):
                    files.append(os.path.join(path, f))
    random.shuffle(files)
    for f in files[:num]:
        os.remove(f)


def resample(path, result_path):
    """
    get 1/8 data
    :param path:
    :return:
    """
    for file in os.listdir(path):
        data = nib.load(os.path.join(path, file)).get_data()
        data = np.squeeze(np.asarray(data))
        data = data[6:-6:2, 6:-6:2, 6:-6:2]
        np.save(os.path.join(result_path, file.replace(".nii", ".npy")), data)


def get_data_from_center(ori_path, result_path):
    # 172 x 220 x 156
    for f in os.listdir(ori_path):
        data = nib.load(os.path.join(ori_path, f)).get_data()
        data = np.asarray(data)
        data = data[10:162, 22:198, 2:154]
        np.save(os.path.join(result_path, f.replace(".nii", ".npy")), data)


def get_dataset_file(paths, result_file):
    """
    Get data set distribution
    :param paths: [train_file, val_file, test_file]
    :param result_file:
    :return:
    """
    f = open(result_file, "a+")
    for path in paths:
        for file in os.listdir(path):
            print(os.path.join(path, file), file=f)
    f.close()


def move_data_according_file(data_path, file):
    f = open(file, "r")
    train_files = []
    val_files = []
    test_files = []
    for line in f.readlines():
        line = line.strip()
        if line.__contains__("train"):
            train_files.append(line.split("/")[-1])
        elif line.__contains__("val"):
            val_files.append(line.split("/")[-1])
        elif line.__contains__("test"):
            test_files.append(line.split("/")[-1])
    for file in os.listdir(data_path):
        if file.endswith(".npy"):
            if file in train_files:
                shutil.move(os.path.join(data_path, file), os.path.join(data_path, "train", file))
            elif file in val_files:
                shutil.move(os.path.join(data_path, file), os.path.join(data_path, "val", file))
            elif file in test_files:
                shutil.move(os.path.join(data_path, file), os.path.join(data_path, "test", file))


def get_val_max_acc(path):
    data = os.system("cat " + path + " | grep val")
    # p = re.compile(r'\d{3}_{0,1}[VS]_{0,1}\d{4}')
    p = re.compile(r'val_acc:.\d.\d{1, 4}')
    val_accs = re.findall(p, data)
    print(val_accs)
    "".split()


def get_ROI_data(path):
    img = nib.load(path).get_data()
    img = np.squeeze(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    # x:y:z= 50:31:50
    # imgs = [img[94:144, 50:81, 94:144], img[94:144, 91:122, 94:144], img[94:144, 127:158, 94:144],
    #         img[21:71, 50:81, 94:144], img[21:71, 91:122, 94:144], img[21:71, 127:158, 94:144]]
    center = get_center(path, 72, 84, 76)

    imgs = [img[center[0]+22:center[0]+72, center[1]-84:center[1]+84, center[2]-76:center[2]+76],
            img[center[0]-51:center[0]-1, center[1]-84:center[1]+84, center[2]-76:center[2]+76],
            img[center[0]-72:center[0]+72, center[1]-34:center[1]-3, center[2]-76:center[2]+76],
            img[center[0]-72:center[0]+72, center[1]+7:center[1]+38, center[2]-76:center[2]+76],
            img[center[0]-72:center[0]+72, center[1]+43:center[1]+74, center[2]-76:center[2]+76],
            img[center[0]-72:center[0]+72, center[1]-84:center[1]+84, center[2]+18:center[2]+68]]
    return imgs


if __name__ == '__main__':
    path = r"C:\Users\fan\Desktop\demo.nii"
    imgs = np.asarray(get_ROI_data(path))
    for i in range(len(imgs)):
        print(imgs[i].shape)
