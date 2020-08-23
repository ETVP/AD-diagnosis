from keras.models import load_model
import numpy as np
from keras import backend as K
import pylab
from skimage.transform import resize
import os
import nibabel as nib
import matplotlib.pyplot as plt
from ori.png2gif import png2gif
from PIL import Image
import cv2


def sensitivity_analysis(img_path, model_path, model_type="conv", relu=True, sigmoid=False):
    # grad-cam
    img_data = np.load(img_path)
    img_data = img_data[np.newaxis, :, :, :, np.newaxis]
    max_ = np.max(img_data)
    min_ = np.min(img_data)
    img_data = (img_data - min_) / (max_ - min_)

    model = load_model(model_path)
    model.summary()
    index = 0
    pred = model.predict(img_data)
    if sigmoid:
        if pred >= 0.5:
            index = 1
    else:
        max_ = np.max(pred)
        for i in range(4):
            if pred[0][i] == max_:
                index = i
            break
    print(pred)

    print("index: ", index)
    pre_output = model.output[:, index]

    if model_type == "conv":
        last_conv_layer = model.get_layer('conv3d_8')
    elif model_type == "unet" or model_type == "unet2":
        last_conv_layer = model.get_layer('conv3d_7')
    elif model_type == "resnet":
        last_conv_layer = model.get_layer("add_3")
        # last_conv_layer = model.get_layer("conv3d_18")
    grads = K.gradients(pre_output, last_conv_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 1, 2, 3))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img_data])

    if relu:
        conv_layer_output_value[np.where(conv_layer_output_value < 0)] = 0
    conv_max = np.max(conv_layer_output_value)
    conv_min = np.min(conv_layer_output_value)
    conv_layer_output_value = (conv_layer_output_value - conv_min) / (conv_max - conv_min)
    pool_max = np.max(pooled_grads_value)
    pool_min = np.min(pooled_grads_value)
    pooled_grads_value = (pooled_grads_value - pool_min) / (pool_max - pool_min)

    layer_number = len(pooled_grads_value)
    for i in range(layer_number):
        conv_layer_output_value[:, :, :, i] *= pooled_grads_value[i]

    # along the last dim calculate the mean value
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    # remove the value which less than 0
    heatmap = np.maximum(heatmap, 0)
    # uniformization
    min_ = np.min(heatmap)
    max_ = np.max(heatmap)
    heatmap = (heatmap - min_) / (max_ - min_)

    return heatmap


def resnet():
    model_path = r'/home/fan/Desktop/ad_classify/log/ADvsNC/T1/resnet2/resnet.loss.ADvsNC.b10.h5'

    img_path = r'/home/fan/Desktop/processed/ADvsNC/train/ad.30.5.npy'
    img_data = np.load(img_path)
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
    heatmap = sensitivity_analysis(img_path=img_path, model_path=model_path, model_type="resnet")

    heatmap = resize(heatmap, (64, 104, 80))

    begin_num = 48
    for i in range(16):
        pylab.subplot(4, 4, i + 1)
        pylab.imshow(img_data[:, :, begin_num + i], cmap='bone')
    pylab.show()
    pylab.figure()

    # pure heatmap
    pylab.axis('off')
    for i in range(16):
        pylab.subplot(4, 4, i + 1)
        pylab.imshow(heatmap[:, :, begin_num + i], cmap='rainbow', vmax=1., vmin=0., alpha=1)
    cbar = pylab.colorbar()
    cbar.ax.tick_params(labelsize=20)
    # end
    pylab.show()


def draw(heatmap, data, save_gif):
    plt.figure()
    plt.axis('off')

    save_path = r"C:\Users\fan\Desktop\occlusion"
    for i in range(len(heatmap)):
        plt.imshow(data[i, :, :], cmap='bone')
        plt.imshow(heatmap[i, :, :], cmap='rainbow', alpha=0.3)
        plt.savefig(os.path.join(save_path, str(i) + '.png'), bbox_inched='tight')

    png2gif(png_path=save_path, save_path=save_gif, num=len(heatmap))

def save_fig(img, dpi, save_path, cmap='rainbow'):
    fig = plt.figure(figsize=(img.shape[1] / dpi, img.shape[0] / dpi), dpi=dpi)
    axes = fig.add_axes([0, 0, 1, 1])
    axes.set_axis_off()
    axes.imshow(img, cmap=cmap)
    fig.savefig(save_path)


def img_fusion(img1, img2, save_path):
    dpi = 100
    save_fig(img1, dpi, "cam.png")
    img = Image.open("cam.png")
    img = np.array(img)

    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j][0] == 127 and img[i][j][1] == 0 and img[i][j][2] == 255 \
                    and img[i][j][3] == 255:
                img[i][j][:] = 255

    save_fig(img2, dpi, "data.png", "bone")
    cam_img = cv2.imread("cam.png")
    data_img = cv2.imread("data.png")
    cam_gray = cv2.cvtColor(cam_img, cv2.COLOR_BGR2GRAY)
    rest, mask = cv2.threshold(cam_gray, 80, 255, cv2.THRESH_BINARY)
    cam_fg = cv2.bitwise_and(cam_img, cam_img, mask=mask)
    dst = cv2.addWeighted(cam_fg, 0.4, data_img, 1, 0)
    add_cubic = cv2.resize(dst, (dst.shape[1] * 4, dst.shape[0] * 4), cv2.INTER_CUBIC)
    cv2.imwrite(save_path, add_cubic)
    print("write over")


if __name__ == '__main__':
    # model_path = r'unet.loss.temp.h5'
    #
    # img_path = r'AD_016_S_0991_MR_MPRAGE_br_raw_20070808103409426_142_S37001_I65803.mniSkull.npy'
    img_path = r"CN_036_S_0813_MR_MPRAGE_br_raw_20060825130911399_1_S18252_I23373.mniSkull.2.npy"
    # #
    # img_data = np.load(img_path)
    # heatmap = sensitivity_analysis(img_path=img_path, model_path=model_path, model_type="unet")
    # heatmap = resize(heatmap, (144, 168, 152))
    # np.save("CNcam.npy", heatmap)
    #
    # # draw(heatmap, img_data, r"C:\Users\fan\Desktop\unet.gif")
    # new_image = nib.Nifti1Image(heatmap, np.eye(4))
    # nib.save(new_image, r'C:\Users\fan\Desktop\my_arr.nii.gz')
    #

    # dpi = 100
    # heatmap = np.load("CNcam.npy")
    # data = np.load(img_path)
    # data = np.asarray(data)
    # heatmap = np.where(heatmap < 0.3, 0, heatmap) * 255
    # for i in range(len(heatmap[0][0])):
    #     img = heatmap[:, :, i]
    #     img = np.asarray(img)
    #     img = np.rot90(img, 1)  # 向左旋转90度
    #     ad_img = data[:, :, i]
    #     ad_img = np.rot90(ad_img, 1)
    #     img_fusion(img, ad_img, r'C:\Users\fan\Desktop\cam2\\'+str(i)+".png")
    # png2gif(png_path=r'C:\Users\fan\Desktop\cam2', save_path=r'C:\Users\fan\Desktop\cam2\cam.gif', num=len(heatmap[0][0]))
    pass


