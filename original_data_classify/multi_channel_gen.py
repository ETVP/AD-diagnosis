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
        for i in range(len(self.files)):
            self.files[i] = self.files[i].replace(".num.0", "").replace(".num.1", "").replace(".num.2", "").\
                replace(".num.3", "").replace(".num.4", "").replace(".num.5", "")
        self.files = list(set(self.files))

    def generate_arrays_from_file(self, batch, sigmoid=False):
        """
        从文件夹得到数据生成器
        :param path:
        :param batch:
        :return:
        """
        while 1:
            count = 0
            label = []
            random.shuffle(self.files)
            for file in self.files:
                label.append(file[0:2])
            label = LabelEncoder().fit_transform(label)
            if not sigmoid:
                label = np_utils.to_categorical(label)
            tmp_label = []

            first_img_data = np.empty([batch, 50, 31, 50])
            second_img_data = np.empty([batch, 50, 31, 50])
            third_img_data = np.empty([batch, 50, 31, 50])
            forth_img_data = np.empty([batch, 50, 31, 50])
            fifth_img_data = np.empty([batch, 50, 31, 50])
            sixth_img_data = np.empty([batch, 50, 31, 50])

            for file in self.files:

                first_img = np.load(os.path.join(self.path, file.replace(".npy", ".num.0.npy")))
                second_img = np.load(os.path.join(self.path, file.replace(".npy", ".num.1.npy")))
                third_img = np.load(os.path.join(self.path, file.replace(".npy", ".num.2.npy")))
                forth_img = np.load(os.path.join(self.path, file.replace(".npy", ".num.3.npy")))
                fifth_img = np.load(os.path.join(self.path, file.replace(".npy", ".num.4.npy")))
                sixth_img = np.load(os.path.join(self.path, file.replace(".npy", ".num.5.npy")))

                first_img_data[count % batch, ] = first_img
                second_img_data[count % batch, ] = second_img
                third_img_data[count % batch, ] = third_img
                forth_img_data[count % batch, ] = forth_img
                fifth_img_data[count % batch, ] = fifth_img
                sixth_img_data[count % batch, ] = sixth_img

                tmp_label.append(label[count])
                count += 1
                if count % batch == 0:

                    first_img_data = first_img_data[:, :, :, :, np.newaxis]
                    second_img_data = second_img_data[:, :, :, :, np.newaxis]
                    third_img_data = third_img_data[:, :, :, :, np.newaxis]
                    forth_img_data = forth_img_data[:, :, :, :, np.newaxis]
                    fifth_img_data = fifth_img_data[:, :, :, :, np.newaxis]
                    sixth_img_data = sixth_img_data[:, :, :, :, np.newaxis]

                    tmp_label = np.asarray(tmp_label)

                    yield ({'input_1': first_img_data, 'input_2': second_img_data, 'input_3': third_img_data,
                            'input_4': forth_img_data, 'input_5': fifth_img_data, 'input_6': sixth_img_data},
                           {'dense_1': tmp_label})
                    tmp_label = []
                    first_img_data = np.empty([batch, 50, 31, 50])
                    second_img_data = np.empty([batch, 50, 31, 50])
                    third_img_data = np.empty([batch, 50, 31, 50])
                    forth_img_data = np.empty([batch, 50, 31, 50])
                    fifth_img_data = np.empty([batch, 50, 31, 50])
                    sixth_img_data = np.empty([batch, 50, 31, 50])
