import random
import numpy as np
import os
import nibabel as nib
from skimage import transform
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from ori.excelExtract import read_excel
import clinical_data_find
from keras.preprocessing.text import Tokenizer


class Generater:

    def __init__(self, path, excel_path):
        self.path = path
        self.files = os.listdir(path)
        self.clinical_data = read_excel(excel_path)
        self.token = Tokenizer(num_words=10000)
        self.token.fit_on_texts(self.clinical_data)

    def generate_arrays_from_file(self, batch, sigmoid=False):
        """
        从文件夹得到数据生成器
        :param path:
        :param batch:
        :return:
        """
        while 1:
            count = 0
            img_data = np.empty([batch, 64, 104, 80])
            label = []
            random.shuffle(self.files)
            for file in self.files:
                label.append(file[0:2])
            label = LabelEncoder().fit_transform(label)
            if not sigmoid:
                label = np_utils.to_categorical(label)
            tmp_label = []
            mmse_datas = []
            sex_datas = []
            age_datas = []
            marriage_datas = []
            apoe4_datas = []
            edu_datas = []
            for file in self.files:
                sub, time = clinical_data_find.name_extract(file)
                clinical_data = clinical_data_find.find_clinical_data(
                    sub, time, self.clinical_data)
                img = np.load(os.path.join(self.path, file))
                img = np.squeeze(img)
                max_ = np.max(img)
                min_ = np.min(img)
                img = (img - min_) / (max_ - min_)
                img_data[count % batch,] = img
                tmp_label.append(label[count])

                mmse = np.asarray([float(clinical_data[0])])
                mmse = mmse[:, np.newaxis]
                mmse_datas.append(mmse)
                sex = 2
                if clinical_data[1] == 'M':
                    sex = 1
                sex = np.asarray([sex])
                sex = sex[:, np.newaxis]
                sex_datas.append(sex)

                age = float(clinical_data[2])
                age = np.asarray([age])
                age = age[:, np.newaxis]
                age_datas.append(age)

                marriage = 0
                if clinical_data[3] == 'Married':
                    marriage = 1
                elif clinical_data[3] == "Widowed":
                    marriage = 2
                elif clinical_data[3] == "Divorced":
                    marriage = 3
                elif clinical_data[3] == "Never married":
                    marriage = 4
                else:
                    marriage = 5
                marriage = np.asarray([marriage])
                marriage = marriage[:, np.newaxis]
                marriage_datas.append(marriage)

                apoe = np.asarray([float(clinical_data[4])])
                apoe = apoe[:, np.newaxis]
                apoe4_datas.append(apoe)

                edu = np.asarray([float(clinical_data[5])])
                edu = edu[:, np.newaxis]
                edu_datas.append(edu)

                count += 1
                if count % batch == 0:
                    img_data = img_data[:, :, :, :, np.newaxis]
                    mmse_datas = np.asarray(mmse_datas)
                    sex_datas = np.asarray(sex_datas)
                    age_datas = np.asarray(age_datas)
                    marriage_datas = np.asarray(marriage_datas)
                    apoe4_datas = np.asarray(apoe4_datas)
                    edu_datas = np.asarray(edu_datas)
                    tmp_label = np.asarray(tmp_label)
                    # Standardization
                    mmse_datas = (mmse_datas - 9) / 21 + 0.01
                    sex_datas = sex_datas / 2 + 0.01
                    age_datas = (age_datas - 56) / 40 + 0.01
                    marriage_datas = marriage_datas / 5 + 0.01
                    apoe4_datas = apoe4_datas / 2 + 0.01
                    edu_datas = (edu_datas - 11) / 9 + 0.01

                    yield ({'input_1': img_data, 'input_2': mmse_datas, 'input_3': sex_datas, 'input_4': age_datas,
                            'input_5': marriage_datas, 'input_6': apoe4_datas, 'input_7': edu_datas},
                           {'dense_1': tmp_label, 'dense_4': tmp_label, 'dense_5': tmp_label})
                    mmse_datas = []
                    sex_datas = []
                    age_datas = []
                    marriage_datas = []
                    apoe4_datas = []
                    edu_datas = []
                    tmp_label = []
                    img_data = np.empty([batch, 64, 104, 80])


def load_predict_img(file):
    """
    从文件夹得到待预测数据
    :param name:
    :return:
    """
    num = len(os.listdir(file))
    img_data = np.empty([num, 256, 256, 166])
    label = []
    for name in os.listdir(file):
        label.append(name[0:2])
    label = LabelEncoder().fit_transform(label)
    i = 0
    for name in os.listdir(file):
        img = nib.load(os.path.join(file, name))
        img = img.get_fdata()
        img = transform.resize(img[:, :, :, 0], (166, 256, 256))
        img = np.squeeze(img)
        img_data[i, ] = img
        max_ = np.max(img_data[i, ])
        min_ = np.min(img_data[i, ])
        img_data[i] = (img_data - min_) / (max_ - min_)
        i += 1
    img_data = img_data[:, :, :, :, np.newaxis]
    return img_data, label
