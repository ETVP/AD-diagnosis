from ori import clinical_main
import model.RnnModel

epoch = 300
batch_size = 50

model = model.RnnModel.Unet().get_model()

loss_file = "log/unet.loss.b5.h5"
acc_file = "log/unet.acc.b5.h5"
draw_file = "convnet.1.b5."
clinical_main.train(model, [loss_file, acc_file, draw_file], epoch, batch_size, False, acc_file)
