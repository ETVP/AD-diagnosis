import matplotlib.pyplot as plt
import imageio
import os


def png2gif(save_path, num, png_path):
    imgs = []
    file_names = []
    for i in range(num):
        file_names.append(str(i) + '.png')
    for file in file_names:
        imgs.append(imageio.imread(os.path.join(png_path, file)))
    imageio.mimsave(save_path, imgs, duration=0.3)


if __name__ == '__main__':
    png_path = "/home/fan/Desktop/heatmap_png"
    png2gif('/home/fan/Desktop/post_img', '/home/fan/Desktop/gif3.gif')
