import numpy as np
import nibabel as nib
import os


path = "/home/fan/Desktop/processed/cn"
result_path = r"F:\AIBL\TestNpy"
data_type = 'cn'


def padding(img):
    newimg = np.zeros((64, 128, 128))
    img_size = img.shape
    offset = (np.array(newimg.shape) - np.array(img_size)) // 2
    newimg[offset[0]: offset[0] + img_size[0],
           offset[1]: offset[1] + img_size[1],
           offset[2]: offset[2] + img_size[2]] = img
    return newimg


def extract(img, min_x=0, max_x=64, min_y=0, max_y=128, min_z=0, max_z=128):
    return img[min_x: max_x, min_y: max_y, min_z: max_z]


def get_x_y_z(img):
    min_x = min_y = min_z = 0
    for i in range(len(img)):
        if img[i].any() == 0:
            min_x = i
        else:
            break
    max_x = len(img) - 1
    for i in range(len(img)-1, -1, -1):
        if img[i].any() == 0:
            max_x = i
        else:
            break

    for j in range(len(img[0])):
        if img[:, j, :].any() == 0:
            min_y = j
        else:
            break
    max_y = len(img[0]) - 1
    for j in range(len(img[0]) - 1, -1, -1):
        if img[:, j, :].any() == 0:
            max_y = j
        else:
            break

    for z in range(len(img[0][0])):
        if img[:, :, z].any() == 0:
            min_z = z
        else:
            break
    max_z = len(img[0][0]) - 1
    for z in range(len(img[0][0]) - 1, -1, -1):
        if img[:, :, z].any() == 0:
            max_z = z
        else:
            break

    if max_x - min_x > 63:
        if (max_x - min_x) % 2 == 0:
            c = (max_x - min_x) / 2 - 31
            max_x -= c
            min_x += c
        else:
            c = int((max_x - min_x) / 2) - 31
            max_x -= c
            min_x += (c + 1)
    elif max_x - min_x < 63:
        c = int((63 - max_x + min_x) / 2)
        l1 = l2 = c
        if c < (63 - max_x + min_x) / 2:
            l2 += 1

        if not (min_x - l1 >= 0 and max_x + l2 < 85):
            if min_x - l1 < 0:
                l2 += (l1-min_x)
                min_x = 0
                max_x += l2
            else:
                l1 += 84 - max_x
                max_x = 84
                min_x -= l1
        else:
            min_x -= l1
            max_x += l2

    if max_y - min_y > 103:
        if (max_y - min_y) % 2 == 0:
            c = int((max_y - min_y) / 2) - 51
            max_y -= c
            min_y += c
        else:
            c = int((max_y - min_y) / 2) - 51
            max_y -= c
            min_y += c + 1
    elif max_y - min_y < 103:
        c = int((103 - max_y + min_y) / 2)
        l1 = l2 = c
        if c < (103 - max_y + min_y) / 2:
            l2 += 1

        if not (min_y - l1 >= 0 and max_y + l2 < 128):
            if min_y - l1 < 0:
                l2 += (l1-min_y)
                min_y = 0
                max_y += l2
            else:
                l1 += 127 - max_y
                max_y = 127
                min_y -= l1
        else:
            min_y -= l1
            max_y += l2

    if max_z - min_z > 79:
        if (max_z - min_z) % 2 == 0:
            c = int((max_z - min_z) / 2) - 39
            max_z -= c
            min_z += c
        else:
            c = int((max_z - min_z) / 2) - 39
            max_z -= c
            min_z += c + 1
    elif max_z - min_z < 79:
        c = int((79 - max_z + min_z) / 2)
        l1 = l2 = c
        if c < (79 - max_z + min_z) / 2:
            l2 += 1
        if not (min_z - l1 >= 0 and max_z + l2 < 128):
            if min_z - l1 < 0:
                l2 += (l1-min_z)
                min_z = 0
                max_z += l2
            else:
                l1 += 127 - max_z
                max_z = 127
                min_z -= l1
        else:
            min_z -= l1
            max_z += l2

    return min_x, max_x, min_y, max_y, min_z, max_z


def save_data(data, i):
    min_x, max_x, min_y, max_y, min_z, max_z = get_x_y_z(data)
    data = extract(img=data, min_x=min_x, max_x=max_x+1, min_y=min_y, max_y=max_y+1,
                   min_z=min_z, max_z=max_z+1)
    # data = padding(data)
    if data.shape != (64, 104, 80):
        print(max_z, " ", min_z)
        print("data error while data shape not is 64*104*80, with the shape ", data.shape)
    # np.save(os.path.join(result_path, data_type + "." + str(i) + ".npy"), data)
    np.save(os.path.join(result_path, file.replace(".nii", ".npy")), data)


if __name__ == '__main__':
    i = 0
    path = r"F:\AIBL\AIBLTest"
    for file in os.listdir(path):
        data = nib.load(os.path.join(path, file)).get_data()
        data = np.asarray(data)
        data = np.squeeze(data)
        # print(data.shape)
        data1 = data[::2, ::2, ::2]
        save_data(data1, i)
        i += 1
        #
        # data2 = data[::2, ::2, 1::2]
        # save_data(data2, i)
        # i += 1
        # #
        # data3 = data[::2, 1::2, ::2]
        # save_data(data3, i)
        # i += 1
        # #
        # data4 = data[::2, 1::2, 1::2]
        # save_data(data4, i)
        # i += 1
        # #
        # data5 = data[1::2, ::2, ::2]
        # save_data(data5, i)
        # i += 1
        # #
        # data6 = data[1::2, ::2, 1::2]
        # save_data(data6, i)
        # i += 1
        # #
        # data7 = data[1::2, 1::2, ::2]
        # save_data(data7, i)
        # i += 1

        # data8 = data[1::2, 1::2, 1::2]
        # save_data(data8, i)
        # i += 1

