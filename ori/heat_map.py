from keras.models import load_model
import numpy as np
from keras import backend as K
import pylab
from skimage.transform import resize
import keras


def cam(model):
    # outlayer = model.get_layer('dense_3')
    activation = model.get_layer('conv3d_8').output


if __name__ == '__main__':
    model_path = r'/home/fan/Desktop/ad_classify/log/ADvsNC/T1/conv4/convnet.loss.2.b10.h5'

    img_path = r'/home/fan/Desktop/processed/ADvsNC/train/ad.0.5.npy'
    model = load_model(model_path)
    cam(model)
