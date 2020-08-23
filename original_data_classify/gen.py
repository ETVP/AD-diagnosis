import random
import numpy as np
import os
import nibabel as nib
from skimage import transform
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


class Generater:

    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)

    def generate_arrays_from_file(self, batch, sigmoid=False):
        """
        从文件夹得到数据生成器
        :param path:
        :param batch:
        :return:
        """
        while 1:
            count = 0
            img_data = np.empty([batch, 120, 168, 136])
            label = []
            random.shuffle(self.files)
            for file in self.files:
                label.append(file[0:2])
            label = LabelEncoder().fit_transform(label)
            if not sigmoid:
                label = np_utils.to_categorical(label)
            tmp_label = []

            for file in self.files:
                img = np.load(os.path.join(self.path, file))
                img = np.asarray(img)
                max_ = np.max(img)
                min_ = np.min(img)
                img = (img - min_) / (max_ - min_)
                img_data[count % batch,] = img
                tmp_label.append(label[count])

                count += 1
                if count % batch == 0:
                    img_data = img_data[:, :, :, :, np.newaxis]
                    yield ({'input_1':img_data, 'dense_1':tmp_label})
                    tmp_label = []
                    img_data = np.empty([batch, 120, 168, 136])
