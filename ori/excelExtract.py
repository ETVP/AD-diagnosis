import pandas as pd
import warnings
import xlsxwriter


warnings.filterwarnings('ignore')


def read_excel(excel_path: str) -> pd.core.frame.DataFrame:
    df = pd.read_excel(excel_path)
    extracted_data = df.ix[:, ["受试者", '图像数据ID', '组别', '性别', '年龄', '获取日期', '受教育程度', '婚姻状况', 'APOE4', '简易精神状态检查表MMSE']]
    # , '问答情况FAQ'
    return extracted_data


def write_excel(excel_path: str, save_data: pd.core.frame.DataFrame):
    # file = xlwt.Workbook()
    # sheet = file.add_sheet('ADClinical', cell_overwrite_ok=True)
    # row0 = ["受试者", '图像ID', '组别', '性别', '年龄', '获取日期', '受教育程度', '婚姻状况', 'APOE4', '简易精神状态检查表MMSE', '问答情况FAQ']

    STYLE_HEADER = {'font_size': 9, 'border': 1, 'bold': 1, 'bg_color': '#B4C6E7', 'align': 'center',
                    'valign': 'vcenter'}
    STYLE_TEXT = {'font_size': 9, 'border': 1}
    STYLE_NUMBER = {'font_size': 9, 'border': 1, 'num_format': '0.00'}
    workbook = xlsxwriter.Workbook(excel_path)
    style_header = workbook.add_format(STYLE_HEADER)
    style_text = workbook.add_format(STYLE_TEXT)
    style_number = workbook.add_format(STYLE_NUMBER)
    AD_sheet = workbook.add_worksheet("ADClinical")
    header = ["受试者", '图像数据ID', '组别', '性别', '年龄', '获取日期', '受教育程度', '婚姻状况', 'APOE4', '简易精神状态检查表MMSE']
    # , '问答情况FAQ'
    AD_sheet.write_row('A1', header, style_header)
    # 宽度设定
    widths = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
    for ind, wid in enumerate(widths):
        AD_sheet.set_column(ind, ind, wid)
    for i in range(len(save_data)):
        AD_sheet.write(i + 1, 0, save_data.iloc[i, 0], style_number)
        AD_sheet.write(i + 1, 1, save_data.iloc[i, 1], style_number)
        AD_sheet.write(i + 1, 2, save_data.iloc[i, 2], style_text)
        AD_sheet.write(i + 1, 3, save_data.iloc[i, 3], style_text)
        AD_sheet.write(i + 1, 4, save_data.iloc[i, 4], style_number)
        AD_sheet.write(i + 1, 5, save_data.iloc[i, 5], style_number)
        AD_sheet.write(i + 1, 6, save_data.iloc[i, 6], style_number)
        AD_sheet.write(i + 1, 7, save_data.iloc[i, 7], style_text)
        AD_sheet.write(i + 1, 8, save_data.iloc[i, 8], style_number)
        AD_sheet.write(i + 1, 9, save_data.iloc[i, 9], style_number)
    workbook.close()


if __name__ == '__main__':
    path = r"D:\data\clinicalData\fromZJ.xlsx"
    write_path = r"D:\data\clinicalData\clinicalData.xlsx"
    data = read_excel(path)
    write_excel(write_path, data)
    # write(write_path)
