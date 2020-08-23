import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# 绘制network fit history
def draw_acc_and_loss_with_val(history, name):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation acc')
    plt.legend()
    plt.savefig("./img/"+ name +"Acc.png")
    # plt.show()

    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("./img/"+ name +"Loss.png")
    # plt.show()


def draw_acc_and_loss(history):
    acc = history.history['acc']
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.title('Training acc')
    plt.legend()
    plt.show()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()
