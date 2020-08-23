from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import backend as K
from keras.preprocessing import image


def activation(img_path, model, layer_name):
    img = np.load(img_path)
    img = img[np.newaxis, :, :, :, np.newaxis]
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    layers_output = model.get_layer(layer_name).output
    activation_model = models.Model(inputs=model.input, outputs=layers_output)
    activations = activation_model.predict(img)
    activations = np.mean(activations, -1)
    activations = np.squeeze(activations)
    plt.axis('off')
    for i in range(int(len(activations[0][0]))):
        plt.imshow(activations[:, :, i], cmap='viridis')
        plt.show()


def filter_visual(img_path, filter_index, layer_name):
    img = np.load(img_path)
    img = img[np.newaxis, :, :, :, np.newaxis]
    img = (img - np.min(img)) /(np.max(img) - np.min(img))

    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([img])
        img += grads_value * step
    img = img[0]
    return img


def draw_layer_filter(layer_name):
    size = 64
    margin = 5
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
    for i in range(8):
        for j in range(8):
            filter_img = filter_visual(img_path, i + (j * 8), layer_name)
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
    plt.figure(figsize=(20, 20))
    results = image.array_to_img(results)
    plt.imshow(results)
    plt.show()


if __name__ == '__main__':
    model_path = r'/home/fan/Desktop/ad_classify/log/NCvsEMvsLMvsAD/unet/unet2.loss.NCvsEMvsLMvsAD.t1.b10.h5'

    img_path = r'/home/fan/Desktop/processed/NCvsEMvsLMvsAD/train/ad.44.3.npy'
    model = load_model(model_path)
    model.summary()
    # activation(img_path, model, 10)
    layer_name = "conv3d_24"
    filter_index = 0
    img = filter_visual(img_path, filter_index, layer_name)
    plt.figure()
    img = np.squeeze(img)
    plt.imshow(img[:, :, 40])
    plt.show()
