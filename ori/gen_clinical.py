import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import pandas as pd


class Generater:

    def __init__(self, excel_path, random_nums):

        df = pd.read_excel(excel_path)
        extracted_data = df.ix[:, ["受试者", '组别', '性别', '年龄', '受教育程度', '婚姻状况', 'APOE4', '简易精神状态检查表MMSE']]
        self.clinical_data = extracted_data
        self.random_nums = random_nums
        # self.token = Tokenizer(num_words=10000)
        # self.token.fit_on_texts(self.clinical_data)

    def generate_arrays_from_file(self, batch, sigmoid=False):
        """
        :param path:
        :param batch:
        :return:
        """
        while 1:
            count = 0
            label = []
            random.shuffle(self.random_nums)
            for num in self.random_nums:
                label.append(self.clinical_data.iloc[num][1])
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
            for num in self.random_nums:
                if self.clinical_data.iloc[num][2] == 'M':
                    sex_datas.append(1)
                else:
                    sex_datas.append(2)
                age_datas.append(float(self.clinical_data.iloc[num][3]))
                edu_datas.append(float(self.clinical_data.iloc[num][4]))

                if self.clinical_data.iloc[num][5] == 'Married':
                    marriage_datas.append(1)
                elif self.clinical_data.iloc[num][5] == "Widowed":
                    marriage_datas.append(2)
                elif self.clinical_data.iloc[num][5] == "Divorced":
                    marriage_datas.append(3)
                elif self.clinical_data.iloc[num][5] == "Never married":
                    marriage_datas.append(4)
                else:
                    marriage_datas.append(5)
                apoe4_datas.append(float(self.clinical_data.iloc[num][6]))
                mmse_datas.append(float(self.clinical_data.iloc[num][7]))
                tmp_label.append(label[count])
                count += 1
                if count % batch == 0:
                    tmp_label = np.array(tmp_label)
                    # convert to numpy
                    mmse_datas = np.asarray(mmse_datas)
                    mmse_datas = (mmse_datas - 9) / 21
                    sex_datas = np.asarray(sex_datas)
                    sex_datas = sex_datas / 2
                    age_datas = np.asarray(age_datas)
                    age_datas = (age_datas - 56) / 40
                    marriage_datas = np.asarray(marriage_datas)
                    marriage_datas = marriage_datas / 5
                    apoe4_datas = np.asarray(apoe4_datas)
                    apoe4_datas = apoe4_datas / 2
                    edu_datas = np.asarray(edu_datas)
                    edu_datas = (edu_datas - 11) / 9
                    yield ({'input_1': mmse_datas, 'input_2': sex_datas, 'input_3': age_datas,
                            'input_4': marriage_datas, 'input_5': apoe4_datas, 'input_6': edu_datas}, {'dense_5': tmp_label})
                    mmse_datas = []
                    sex_datas = []
                    age_datas = []
                    marriage_datas = []
                    apoe4_datas = []
                    edu_datas = []
                    tmp_label = []
