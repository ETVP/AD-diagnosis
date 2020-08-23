from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import Model
import os


def layer_learned(model: Model, layer_name, img_path):
    img = np.load(img_path)
    img = img[np.newaxis, :, :, :, np.newaxis]
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    layer_output = model.get_layer(layer_name).output
    activation_mode = models.Model(inputs=model.input, outputs=layer_output)
    activations = activation_mode.predict(img)
    activations = np.mean(activations, -1)
    activations = np.squeeze(activations)
    plt.figure()

    for i in range(9):
        plt.subplot(3, 3, 1+i)
        plt.imshow(activations[:, :, 20+i], cmap='viridis')
    plt.savefig(os.path.join(pic_path, 'second_conv_learned.png'))
    plt.show()


if __name__ == '__main__':
    pic_path = '/home/fan/Desktop/result_pic'
    # model_path = r'/home/fan/Desktop/ad_classify/log/ADvsNC/T1/conv4/convnet.loss.2.b10.h5'
    img_path = r'ad.100.5.npy'
    layer_name = "conv3d_62"
    # model = load_model(model_path)
    # model.summary()
    # layer_learned(model, layer_name, img_path)
    model_path = r'/home/fan/Desktop/ad_classify/log/ADvsNC/T1/resnet/resnet.loss.ADvsNC.t1.b10.h5'
    model = load_model(model_path)
    model.summary()
    # layer_name = "add_2"
    layer_learned(model, layer_name, img_path)
