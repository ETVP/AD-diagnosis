from ori import main
import combine


epoch = 300
batch_size = 5

model = combine.Unet((64, 104, 80, 1), (168, 184, 48, 1)).get_model()
img_file = r"C:\Users\fan\Desktop\AD\newSH"
loss_file = "log/unet.loss.b5.h5"
acc_file = "log/unet.acc.b5.h5"
draw_file = "convnet.1.b5."
main.train(img_file, model, [loss_file, acc_file, draw_file], epoch, batch_size, True)
