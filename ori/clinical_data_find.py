import pandas as pd
import warnings
from ori.excelExtract import read_excel
import os


warnings.filterwarnings('ignore')


def name_extract(mri_name: str):
    """
    从MRI文件名中提起subject和time,并整理成对应格式
    subject: ***_S_***
    time: [year, month]  这里我们不需要日期也是一致的
    :param mri_name:
    :return:
    """
    ss = mri_name.split(".")
    subject = ss[1]
    time = ss[2]
    subjects = subject.split("S")
    subject = subjects[0] + '_S_' + subjects[1]
    time = [int(time[:4]), int(time[4:6])]
    return subject, time


def data_frame_to_list(ori_data: pd.core.frame.DataFrame) -> list:
    """
    将DataFrame类型数据转为List数据
    :param ori_data:
    :return:
    """
    result = []
    for i in range(len(ori_data)):
        result.append(ori_data[i])
    return result


def find_clinical_data(subject: str, acq_date: list, clinical_data: pd.core.frame.DataFrame):
    """
    通过subject和acq_data获取对应的病人临床信息
    :param subject:
    :param acq_date: 图像获取时间
    :param clinical_data: 从excel文件中读取的临床数据，数据的第一个属性为受试者->subject，第二个属性为获取日期->acq_data
    :return:
    """
    subjects = clinical_data.ix[:, ["受试者", "获取日期"]]
    for i in range(len(clinical_data)):
        if subjects.iloc[i, 0] == subject:
            year = acq_date[0]
            month = acq_date[1]
            data_date = subjects.iloc[i, 1]
            data_dates = data_date.split("/")
            year1 = int(data_dates[2])
            month1 = int(data_dates[0])
            if year == year1 and (month == month1 or month == month1 - 1 or month == month1 + 1):
                # print(clinical_data.ix[i, ['性别', '年龄', '受教育程度', '婚姻状况', 'APOE4', '简易精神状态检查表MMSE']])
                # return clinical_data.ix[i, ['性别', '年龄', '受教育程度', '婚姻状况', 'APOE4', '简易精神状态检查表MMSE']]
                # print(clinical_data.ix[i, ['简易精神状态检查表MMSE']][0])
                return clinical_data.ix[i, ['简易精神状态检查表MMSE', '性别', '年龄', '婚姻状况', 'APOE4', '受教育程度']]
    print("not find for subject ", subject, " and time ", acq_date)
    return None


def find_test(mri_path: str, excel_path: str):
    data = read_excel(excel_path)
    files = os.listdir(mri_path)
    for file in files:
        sub, time = name_extract(file)
        find_clinical_data(sub, time, data)


def get_cli(excel_path):
    df = pd.read_excel(excel_path)
    extracted_data = df.ix[:, ["Sex", 'Research Group', 'APOE A1', 'APOE A2', 'Age', 'GDSCALE Total Score',
                               'Global CDR', 'FAQ Total Score', 'NPI-Q Total Score']]
    data = []
    for i in range(len(extracted_data)):
        tmp = []
        for j in range(len(extracted_data.ix[i])):
            tmp.append(extracted_data.ix[i][j])
        data.append(tmp)
    # for item in data:
    #     print(item)
    return data


if __name__ == '__main__':
    get_cli(r"F:\ADNIADMPRAGESIEMENS\fixed_cli_score.xlsx")

