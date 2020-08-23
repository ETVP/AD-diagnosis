import numpy as np
import matplotlib.pyplot as plt
import os
from ori.png2gif import png2gif
from keras import Model
from keras.models import load_model


def occlusion(img_path, model:Model, stride=7, step=3):
    """
    3D块遮挡
    :param img_path:
    :param model:
    :param stride: 遮挡半径
    :param step: 延伸距离
    :return:
    """
    img_data = np.load(img_path)
    img_data = img_data[np.newaxis, :, :, :, np.newaxis]

    pred = model.predict(img_data)

    heatmap = np.zeros(shape=(144, 168, 152))
    number = 0
    total_number = int(len(img_data[0]) / step * len(img_data[0][0]) / step * len(img_data[0][0][0]) / step)
    pix_number = (stride*2+step)*(stride*2+step)*(stride*2+step)

    x_max = len(img_data[0])
    y_max = len(img_data[0][0])
    z_max = len(img_data[0][0][0])
    for x in range(0, len(img_data[0]), step):
        for y in range(0, len(img_data[0][0]), step):
            for z in range(0, len(img_data[0][0][0]), step):
                print("number: ", number, " total: ", total_number)
                x_from = max(x-stride, 0)
                x_to = min(x+stride+1, x_max)
                y_from = max(y-stride, 0)
                y_to = min(y+stride+1, y_max)
                z_from = max(z-stride, 0)
                z_to = min(z+stride+1, z_max)

                tmp_data = img_data.copy()

                tmp_data[:, x_from:x_to+1, y_from:y_to+1, z_from:z_to+1, :] = 0
                tmp_output = model.predict(tmp_data)

                heatmap[x:x+step, y:y+step, z:z+step] = (abs(pred[0][0] - tmp_output[0][0]) / pix_number)

                number += 1

    return heatmap


def big_occlusion(img_path, model:Model, x_y=(8, 8)):
    """
    2D块遮挡
    :param img_path:
    :param model:
    :param x_y:
    :return:
    """
    img = np.load(img_path)
    img = img[np.newaxis, :, :, :, np.newaxis]
    pred = model.predict(img)
    num = 0
    print(int(len(img[0]) / x_y[0]), " ", int(len(img[0][0]) / x_y[1]), " ", len(img[0][0][0]))
    total = int(len(img[0]) / x_y[0]) * int(len(img[0][0]) / x_y[1]) * len(img[0][0][0])
    heatmap = np.zeros(shape=(64, 104, 80))
    pix_num = x_y[0] * x_y[1]
    for i in range(len(img[0][0][0])):
        for j in range(0, len(img[0][0]), x_y[1]):
            for k in range(0, len(img[0]), x_y[0]):
                num += 1
                print("num: ", num, " total: ", total)
                tmp_img = img.copy()
                tmp_img[:, k:k+x_y[0], j:j+x_y[1], i, :] = 0
                tmp_pre = model.predict(tmp_img)
                heatmap[k:k+x_y[0], j:j+x_y[1], i] = abs((pred[0][0] - tmp_pre[0][0])) / pix_num
    return heatmap


def isoheight(img_path, model, num=1000):
    """
    灰度值区间遮挡
    :param img_path:
    :param model:
    :param num:灰度值区间划分份数
    :return:
    """
    img_data = np.load(img_path)
    # ori_img_data = img_data[np.newaxis, :, :, :, np.newaxis]
    ori_img_data = img_data.copy()
    ori_img_data = ori_img_data[np.newaxis, :, :, :, np.newaxis]
    pred = model.predict(ori_img_data)

    presion = 1 / num
    heatmap = np.zeros(shape=(144, 168, 152))
    height_high = 0
    for z in range(num):
        height_low = height_high
        height_high += presion
        tmp_data = img_data.copy()
        min_h = np.where(tmp_data > height_low, 0, 1)
        max_h = np.where(tmp_data < height_high, 0, 1)
        mix_h = min_h + max_h
        tmp_data = np.where(mix_h, tmp_data, 0)
        tmp_data = tmp_data[np.newaxis, :, :, :, np.newaxis]
        tmp_pre = model.predict(tmp_data)
        print("number ", z, " total: ", num, " dis: ", (tmp_pre[0][0] - pred[0][0]))
        dis = tmp_pre[0][0] - pred[0][0]
        number = 144*168*152-np.sum(mix_h)
        heatmap = heatmap + np.where(mix_h, heatmap, dis/number)

    return heatmap


def multi_isoheight(img_path, model, size=(100, 1000)):
    """
    混合多灰度值区间遮挡
    :param img_path:
    :param model:
    :param size:
    :return:
    """
    heatmap = np.zeros(shape=(64, 104, 80))
    rate = []
    if len(size) == 2:
        rate.append(0.3)
        rate.append(0.7)
        heatmap1 = isoheight(img_path, model, size[0])
        heatmap2 = isoheight(img_path, model, size[1])
        heatmap = heatmap1 * rate[0] + heatmap2 * rate[1]
    elif len(size) == 3:
        # bad 10, 100, 1000
        rate.append(0.1)
        rate.append(0.3)
        rate.append(0.6)
        heatmap1 = isoheight(img_path, model, size[0])
        heatmap2 = isoheight(img_path, model, size[1])
        heatmap3 = isoheight(img_path, model, size[2])
        heatmap = heatmap1 * rate[0] + heatmap2 * rate[1] + heatmap3 * rate[2]
    return heatmap


def draw(heatmap, save_gif, colorbar=False, dim=2, save_path=""):
    """
    按照dim绘制热力图
    :param heatmap:
    :param save_gif:gif保存路径
    :param colorbar:
    :param dim:
    :param save_path: 中间png图像保存路径
    :return:
    """
    plt.axis('off')
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

    if dim == 2:
        for i in range(len(heatmap[0][0])):
            plt.figure()
            plt.imshow(heatmap[:, :, i], cmap='rainbow')
            if colorbar:
                plt.colorbar()
            plt.savefig(os.path.join(save_path, str(i) + '.png'), bbox_inched='tight')
        png2gif(save_path=save_gif, num=len(heatmap[0][0]), png_path=save_path)
    elif dim == 1:
        for i in range(len(heatmap[0])):
            plt.figure()
            plt.imshow(heatmap[:, i, :], cmap='rainbow')
            if colorbar:
                plt.colorbar()
            plt.savefig(os.path.join(save_path, str(i) + '.png'), bbox_inched='tight')
        png2gif(save_path=save_gif, num=len(heatmap[0]), png_path=save_path)
    elif dim == 0:
        for i in range(len(heatmap)):
            plt.figure()
            plt.imshow(heatmap[i, :, :], cmap='rainbow')
            if colorbar:
                plt.colorbar()
            plt.savefig(os.path.join(save_path, str(i) + '.png'), bbox_inched='tight')
        png2gif(save_path=save_gif, num=len(heatmap), png_path=save_path)
    else:
        print("dim erreo")
        return


def avg_occlusion(img_path, model):
    """
    对配准后的图像集进行使用
    :param img_path:
    :param model:
    :return:
    """
    heatmap = np.zeros(shape=(64, 104, 80))
    for top, dirs, nondirs in os.walk(img_path):
        for name in nondirs:
            if name.__contains__("ad"):
                heatmap += occlusion(os.path.join(top, name), model=model)
    heatmap = (heatmap - np.min(heatmap) / (np.max(heatmap) - np.min(heatmap)))
    np.save('avg_ad_heatmap.npy', heatmap)


def draw_one(heatmap_path, ori_path, slice):
    """
    绘制一个切片
    :param heatmap_path:
    :param ori_path:
    :param slice:
    :return:
    """
    heatmap = np.load(heatmap_path)
    data = np.load(ori_path)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    plt.axis('off')

    plt.imshow(data[:, :, slice], cmap=plt.cm.Greys_r)
    plt.imshow(heatmap[:, :, slice], cmap=plt.cm.rainbow, alpha=0.3)
    plt.show()


def add_draw(data_path, heatmap_path, save_file, gif_path, dim=2):
    """
    绘制热力图叠加原图后的图像
    :param data_path:
    :param heatmap_path:
    :param save_file:
    :param gif_path:
    :param dim:
    :return:
    """
    heatmap = np.load(heatmap_path)
    data = np.load(data_path)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    plt.axis('off')
    if dim == 2:
        for i in range(len(data[0][0])):
            plt.imshow(heatmap[:, :, i], cmap='rainbow')
            plt.imshow(data[:, :, i], cmap='bone', alpha=0.6)
            plt.savefig(os.path.join(save_file, str(i) + '.png'), bbox_inched='tight')
        png2gif(save_path=gif_path, num=len(heatmap[0][0]), png_path=save_file)
    elif dim == 1:
        for i in range(len(data[0])):
            plt.imshow(heatmap[:, i, :], cmap='rainbow')
            plt.imshow(data[:, i, :], cmap='bone', alpha=0.6)
            plt.savefig(os.path.join(save_file, str(i) + '.png'), bbox_inched='tight')
        png2gif(save_path=gif_path, num=len(heatmap[0]), png_path=save_file)
    elif dim == 0:
        for i in range(len(data)):
            plt.imshow(heatmap[i, :, :], cmap='rainbow')
            plt.imshow(data[i, :, :], cmap='bone', alpha=0.6)
            plt.savefig(os.path.join(save_file, str(i) + '.png'), bbox_inched='tight')
        png2gif(save_path=gif_path, num=len(heatmap), png_path=save_file)
    else:
        print("dim error")
        return


if __name__ == '__main__':
    model_path = r'unet.loss.temp.h5'
    model = load_model(model_path)
    ad_img_path = 'AD_016_S_0991_MR_MPRAGE_br_raw_20070808103409426_142_S37001_I65803.mniSkull.npy'

    occ_path = r'/home/fan/Desktop/occlusion/iso.gif'
    save_png = r'/home/fan/Desktop/occlusion'
    heatmap = isoheight(ad_img_path, model, num=1000)
    np.save('ad.conv.iso.1000.npy', heatmap)
    draw(heatmap, save_gif=occ_path, colorbar=False, dim=2, save_path=save_png)
    # heatmap = big_occlusion(img_path, model=model, x_y=(8, 13))
    # heatmap = multi_isoheight(img_path, model)
    # np.save('ad.conv.occ.npy', heatmap)
    # draw(heatmap, save_gif=occ_path, colorbar=False, dim=2, save_path=save_png)
    # heatmap_path = '/home/fan/Desktop/heatmap/occlusion/mul-iso/conv_cite100-1000/ad.conv.occ.npy'
    add_draw(ad_img_path, 'ad.conv.iso.1000.npy', save_png, occ_path, dim=1)
    # heatmap = occlusion(ad_img_path, model)
    # np.save('ad.conv.iso.2.3.npy', heatmap)
    # draw(heatmap, save_gif=occ_path, colorbar=False, dim=2, save_path=save_png)
