import sys
from keras.models import load_model
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QPalette
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import os
from skimage.transform import resize
from ori import preprocess


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        self.appName = 'AD predictor'
        self.picture = r'./img/ad_tmp.nii'
        self.img_path = r'./img/ad_tmp.nii'
        self.picture1 = ""
        self.picture2 = ""
        self.heatmap_pic = ""
        self.img_data = None
        self.model = "NC vs AD"
        self.NCvsADModel = "models/ADvsNC.h5"
        self.NCvsEMModel = "models/NCvsEM.h5"
        self.EMvsLMModel = "models/EMvsLM.h5"
        self.LMvsADModel = "models/ADvsLM.h5"
        self.threeClassModel = "models/NCvsEMvsLM.h5"
        self.fourClassModel = "models/NCvsEMvsLMvsAD.h5"
        self.open_file = "./"

        img_data = nib.load(self.img_path).get_data()
        z_min, z_max, x_min, x_max, y_min, y_max = self.blank_filter(img_data)

        plt.axis('off')

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        tmp_img = img_data[:, :, int((z_min + z_max) / 2)]
        tmp_img = self.flip90_right(tmp_img)

        # print('max: ', np.max(tmp_img))
        # print('min: ', np.min(tmp_img))
        pix_dis = np.max(tmp_img) - np.min(tmp_img)
        tmp_img[np.where(tmp_img < pix_dis/5)] = 0

        plt.imshow(tmp_img, cmap='bone')
        plt.savefig("tmp.jpg", bbox_inches='tight')

        self.picture = "tmp.jpg"

        super(MainWindow, self).__init__(parent)

        self.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.white)
        self.setPalette(palette)

        # self.resize(800, 500)
        self.setFixedSize(800, 560)
        self.center()
        self.status = self.statusBar()
        self.setWindowTitle(self.appName)

        bar = self.menuBar()

        bar.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.white)
        bar.setPalette(palette)

        file = bar.addMenu("File")
        file.addAction("Open")
        save = QAction("Save", self)
        save.setShortcut("Ctrl+S")
        file.addAction(save)
        quit = QAction("Exit", self)
        file.addAction(quit)
        help = bar.addMenu("Help")
        help.addAction("How to use it")
        help.addAction("About " + self.appName)
        file.triggered[QAction].connect(self.process_trigger)

        welcome_label = QLabel(self)
        welcome_label.setAutoFillBackground(True)
        # palette = QPalette()
        # palette.setColor(QPalette.Window, Qt.white)
        welcome_label.setPalette(palette)

        font_family = "Arial"
        font = QFont()
        font.setFamily(font_family)
        font.setPointSize(30)
        font.setBold(True)
        font.setWeight(75)

        welcome_label.setFont(font)
        welcome_label.setFixedSize(400, 200)
        welcome_label.setText("<font color=GreenYellow>Welcome</font>")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.move(0, 20)

        self.slide1 = QSlider(Qt.Horizontal, self)
        self.slide1.setMinimum(z_min)
        self.slide1.setMaximum(z_max)
        self.slide1.setValue(0)
        self.slide1.setSingleStep(1)
        self.slide1.setAutoFillBackground(True)
        self.slide1.setPalette(palette)
        self.slide1.setFixedSize(140, 30)
        self.slide1.setTickInterval(5)
        self.slide1.setTickPosition(QSlider.TicksBelow)
        self.slide1.move(420, 20)
        self.slide1.valueChanged.connect(self.slide_event1)

        self.slide2 = QSlider(Qt.Horizontal, self)
        self.slide2.setMinimum(x_min)
        self.slide2.setMaximum(x_max)
        self.slide2.setValue(0)
        self.slide2.setSingleStep(1)
        self.slide2.setAutoFillBackground(True)
        self.slide2.setPalette(palette)
        self.slide2.setFixedSize(140, 30)
        self.slide2.setTickInterval(5)
        self.slide2.setTickPosition(QSlider.TicksBelow)
        self.slide2.move(420, 280)
        self.slide2.valueChanged.connect(self.slide_event2)

        self.slide3 = QSlider(Qt.Horizontal, self)
        self.slide3.setMinimum(y_min)
        self.slide3.setMaximum(y_max)
        self.slide3.setValue(0)
        self.slide3.setSingleStep(1)
        self.slide3.setAutoFillBackground(True)
        self.slide3.setPalette(palette)
        self.slide3.setFixedSize(140, 30)
        self.slide3.setTickInterval(5)
        self.slide3.setTickPosition(QSlider.TicksBelow)
        self.slide3.move(620, 280)
        self.slide3.valueChanged.connect(self.slide_event3)

        load_data_bt = QPushButton(self)
        load_data_bt.setText("Load Data")
        load_data_bt.setFixedSize(100, 30)
        load_data_bt.move(100, 190)
        load_data_bt.clicked.connect(self.load_data_event)

        predict_bt = QPushButton(self)
        predict_bt.setText("Predict")
        predict_bt.setFixedSize(100, 30)
        predict_bt.move(220, 190)
        predict_bt.clicked.connect(self.predict_button_event)

        heatmap_bt = QPushButton(self)
        heatmap_bt.setText("Heatmap")
        heatmap_bt.setFixedSize(100, 30)
        heatmap_bt.move(100, 230)

        black_palette = QPalette()
        black_palette.setColor(QPalette.Window, Qt.black)

        self.size = QSize(210, 280)

        self.back1 = QLabel(self)
        self.back1.setAlignment(Qt.AlignCenter)
        self.back1.move(400, 40)
        self.back1.setStyleSheet("background-color:black;")
        # self.result_label.setPalette(palette)
        self.back1.setFixedSize(180, 240)

        self.back2 = QLabel(self)
        self.back2.setAlignment(Qt.AlignCenter)
        self.back2.move(400, 300)
        self.back2.setStyleSheet("background-color:black;")
        self.back2.setFixedSize(180, 240)

        self.back3 = QLabel(self)
        self.back3.setAlignment(Qt.AlignCenter)
        self.back3.move(600, 300)
        self.back3.setStyleSheet("background-color:black;")
        self.back3.setFixedSize(180, 240)

        self.img_label = QLabel(self)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setPixmap(QPixmap(self.picture).scaled(self.size))

        self.img_label.setFixedSize(180, 240)
        self.img_label.setVisible(False)
        self.img_label.move(400, 40)
        # self.img_label.setAcceptDrops(True)
        self.setAcceptDrops(True)
        # self.setAcceptDrops(True)

        heatmap_style = '''
        background-color:black
        '''
        self.heatmap = QLabel(self)
        self.heatmap.setAlignment(Qt.AlignCenter)
        self.heatmap.setPixmap(QPixmap(self.heatmap_pic).scaled(self.size))
        self.heatmap.setFixedSize(180, 240)
        self.heatmap.setStyleSheet(heatmap_style)
        self.heatmap.move(600, 40)

        self.img_label2 = QLabel(self)
        self.img_label2.setAlignment(Qt.AlignCenter)
        self.img_label2.setPixmap(QPixmap(self.picture1).scaled(self.size))
        self.img_label2.setFixedSize(180, 240)
        self.img_label2.setPalette(black_palette)
        self.img_label2.setVisible(True)
        self.img_label2.move(400, 300)

        self.img_label3 = QLabel(self)
        self.img_label3.setAlignment(Qt.AlignCenter)
        self.img_label3.setPixmap(QPixmap(self.picture2).scaled(self.size))
        self.img_label3.setFixedSize(180, 240)
        self.img_label3.setPalette(black_palette)
        self.img_label3.setVisible(True)
        self.img_label3.move(600, 300)

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.move(40, 270)
        self.result_label.setStyleSheet("background-color:lightGrey;")
        self.result_label.setFixedSize(320, 200)

        font1 = QFont()
        font1.setFamily(font_family)
        font1.setPointSize(15)
        font1.setBold(False)
        font1.setWeight(35)

        self.result_title = QLabel(self)
        self.result_title.setText("The Outcome of Predict")
        self.result_title.setFont(font1)
        self.result_title.setFixedSize(260, 30)
        self.result_title.setAlignment(Qt.AlignCenter)
        self.result_title.move(70, 280)

        font2 = QFont()
        font2.setFamily(font_family)
        font2.setPointSize(14)
        font2.setBold(False)
        # font2.setWeight(20)

        self.nc_label = QLabel(self)
        self.nc_label.setAlignment(Qt.AlignCenter)
        self.nc_label.setText(" NC ")
        self.nc_label.setFixedSize(50, 30)
        self.nc_label.move(50, 330)
        self.nc_label.setFont(font2)

        self.emci_label = QLabel(self)
        self.emci_label.setAlignment(Qt.AlignCenter)
        self.emci_label.setText("EMCI")
        self.emci_label.setFixedSize(50, 30)
        self.emci_label.move(50, 360)
        self.emci_label.setFont(font2)

        self.lmci_label = QLabel(self)
        self.lmci_label.setAlignment(Qt.AlignCenter)
        self.lmci_label.setText("LMCI")
        self.lmci_label.setFixedSize(50, 30)
        self.lmci_label.move(50, 390)
        self.lmci_label.setFont(font2)

        self.ad_label = QLabel(self)
        self.ad_label.setAlignment(Qt.AlignCenter)
        self.ad_label.setText(" AD ")
        self.ad_label.setFixedSize(50, 30)
        self.ad_label.move(50, 420)
        self.ad_label.setFont(font2)

        self.combo = QComboBox(self)
        self.combo.addItem("NC vs AD")
        self.combo.addItem("NC vs EM")
        self.combo.addItem("EM vs LM")
        self.combo.addItem("LM vs AD")
        self.combo.addItem("NC vs EM vs LM")
        self.combo.addItem("NC vs EM vs LM vs AD")
        self.combo.setFixedSize(100, 30)
        self.combo.currentTextChanged.connect(self.comboEvent)
        self.combo.move(220, 230)

        gray_palette = QPalette()
        gray_palette.setColor(QPalette.Window, Qt.gray)

        self.nc_probability = QProgressBar(self)
        self.nc_probability.setGeometry(150, 332, 200, 26)
        self.nc_probability.setMinimum(0)
        self.nc_probability.setMaximum(100)
        self.nc_probability.setValue(0)
        self.nc_probability.setTextVisible(False)

        nc_style = '''
                    QProgressBar{background-color: Gainsboro; text-align:center}
                    QProgressBar::chunk{background-color: green}
                '''
        self.nc_probability.setStyleSheet(nc_style)

        self.em_probability = QProgressBar(self)
        self.em_probability.setGeometry(150, 362, 200, 26)
        self.em_probability.setMinimum(0)
        self.em_probability.setMaximum(100)
        self.em_probability.setValue(0)
        self.em_probability.setTextVisible(False)

        em_style = '''
                    QProgressBar{background-color: Gainsboro; text-align:center}
                    QProgressBar::chunk{background-color: blue}
                '''
        self.em_probability.setStyleSheet(em_style)

        self.lm_probability = QProgressBar(self)
        self.lm_probability.setGeometry(150, 392, 200, 26)
        self.lm_probability.setMinimum(0)
        self.lm_probability.setMaximum(100)
        self.lm_probability.setValue(0)
        self.lm_probability.setTextVisible(False)

        lm_style = '''
                    QProgressBar{background-color: Gainsboro; text-align:center}
                    QProgressBar::chunk{background-color: yellow}
                '''
        self.lm_probability.setStyleSheet(lm_style)

        self.ad_probability = QProgressBar(self)
        self.ad_probability.setGeometry(150, 422, 200, 26)
        self.ad_probability.setMinimum(0)
        self.ad_probability.setMaximum(100)
        self.ad_probability.setValue(0)

        ad_style = '''
                    QProgressBar{background-color: Gainsboro; text-align:center}
                    QProgressBar::chunk{background-color: red}
                  '''
        self.ad_probability.setTextVisible(False)
        self.ad_probability.setStyleSheet(ad_style)

    def get_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open File', self.open_file, "nifty image (*.nii)")
        return fname

    def dropEvent(self, evn):
        pass

    def comboEvent(self):
        sender = self.sender()
        self.model = sender.currentText()

    def process_trigger(self, q):
        if q.text() == "Exit":
            quit_dialog = QMessageBox.question(self, "Confirm Exit",
                                               "Are you sure you want to exit " + self.appName,
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if quit_dialog == QMessageBox.Yes:
                sys.exit(app.exec_())
        elif q.text() == "Open":
            self.load_data_event()

    def load_data_event(self):
        f_name = self.get_file()
        self.img_path = f_name
        self.open_file = os.path.abspath(os.path.dirname(f_name)+os.path.sep+"./")

        if not os.path.exists(self.img_path):
            return

        self.img_data = nib.load(self.img_path).get_data()
        z_min, z_max, x_min, x_max, y_min, y_max = self.blank_filter(self.img_data)

        self.slide1.setMinimum(z_min)
        self.slide1.setMaximum(z_max)
        self.slide1.setValue(int((z_min + z_max) / 2))

        self.img_label.setVisible(True)
        # self.file_label.setText("<font color=white>" + self.img_path.split("/")[-1] + "</font>")

        self.slide2.setMinimum(x_min)
        self.slide2.setMaximum(x_max)
        self.slide2.setValue(int((x_min + x_max) / 2))
        self.img_label2.setVisible(True)

        self.slide3.setMinimum(y_min)
        self.slide3.setMaximum(y_max)
        print("y_min ", y_min, ' y_max ', y_max)
        self.slide3.setValue(int((y_min + y_max) / 2))
        self.img_label3.setVisible(True)
        self.predict_clear()

    def predict_clear(self):
        self.nc_probability.setTextVisible(False)
        self.nc_probability.setValue(0)
        self.ad_probability.setTextVisible(False)
        self.ad_probability.setValue(0)
        self.em_probability.setValue(0)
        self.em_probability.setTextVisible(False)
        self.lm_probability.setTextVisible(False)
        self.lm_probability.setValue(0)

    def slide_event1(self):
        try:
            sender = self.sender()
            slice = int(sender.value())
            # self.img_data = nib.load(self.img_path).get_data()

            tmp_img = self.flip90_right(self.img_data[:, :, slice])
            pix_dis = np.max(tmp_img) - np.min(tmp_img)
            tmp_img[np.where(tmp_img < pix_dis / 5)] = 0

            plt.imshow(tmp_img, cmap='bone')
            plt.savefig("tmp.jpg", bbox_inches='tight')

            self.picture = "tmp.jpg"
            self.img_label.setPixmap(QPixmap(self.picture).scaled(self.size))
        except:
            pass

    def slide_event2(self):
        try:
            sender = self.sender()
            slice = int(sender.value())

            tmp_img = self.img_data[slice, :, :]
            # tmp_img = self.flip90_right(img_data[slice, :, :])
            pix_dis = np.max(tmp_img) - np.min(tmp_img)
            tmp_img[np.where(tmp_img < pix_dis / 5)] = 0

            tmp_img = resize(tmp_img, (180, 240))

            plt.imshow(tmp_img, cmap='bone')
            plt.savefig('tmp1.jpg', bbox_inched='tight')
            self.picture1 = 'tmp1.jpg'
            self.img_label2.setPixmap(QPixmap(self.picture1).scaled(self.size))
        except:
            pass

    def slide_event3(self):
        try:
            sender = self.sender()
            slice = int(sender.value())
            tmp_img = self.img_data[:, slice, :]
            pix_dis = np.max(tmp_img) - np.min(tmp_img)
            tmp_img[np.where(tmp_img < pix_dis / 5)] = 0
            tmp_img = self.flip90_left(tmp_img)

            tmp_img = resize(tmp_img, (180, 240))

            plt.imshow(tmp_img, cmap='bone')
            plt.savefig('tmp2.jpg', bbox_inched='tight')
            self.picture2 = 'tmp2.jpg'
            self.img_label3.setPixmap(QPixmap(self.picture2).scaled(self.size))
        except:
            pass

    def predict_button_event(self):
        model = None
        data = np.asarray(self.img_data)
        data = np.squeeze(data)
        data = data[::2, ::2, ::2]
        min_x, max_x, min_y, max_y, min_z, max_z = preprocess.get_x_y_z(data)
        data = preprocess.extract(img=data, min_x=min_x, max_x=max_x + 1, min_y=min_y, max_y=max_y + 1,
                                  min_z=min_z, max_z=max_z + 1)
        data = data[np.newaxis, :, :, :, np.newaxis]

        data = (data - np.min(data)) / (np.max(data) - np.min(data))

        self.predict_clear()
        if self.model == "NC vs AD":
            model = load_model(self.NCvsADModel)
            predict = model.predict(data)
            print(predict)
            self.ad_probability.setValue(round(100 - predict[0][0] * 100, 2))
            self.ad_probability.setTextVisible(True)
            self.nc_probability.setValue(round(predict[0][0] * 100, 2))
            self.nc_probability.setTextVisible(True)
        elif self.model == "NC vs EM":
            model = load_model(self.NCvsEMModel)
            predict = model.predict(data)
            self.nc_probability.setValue(round(100-predict[0][0] * 100, 2))
            self.nc_probability.setTextVisible(True)
            self.em_probability.setValue(round(predict[0][0] * 100, 2))
            self.em_probability.setTextVisible(True)
        elif self.model == "EM vs LM":
            model = load_model(self.EMvsLMModel)
            predict = model.predict(data)
            self.em_probability.setValue(round(100-predict[0][0] * 100, 2))
            self.em_probability.setTextVisible(True)
            self.lm_probability.setValue(round(predict[0][0] * 100, 2))
            self.lm_probability.setTextVisible(True)
        elif self.model == "LM vs AD":
            model = load_model(self.LMvsADModel)
            predict = model.predict(data)
            self.ad_probability.setValue(round(100-predict[0][0] * 100, 2))
            self.ad_probability.setTextVisible(True)
            self.lm_probability.setValue(round(predict[0][0] * 100, 2))
            self.lm_probability.setTextVisible(True)
        elif self.model == "NC vs EM vs LM":
            model = load_model(self.threeClassModel)
            predict = model.predict(data)
            self.nc_probability.setValue(round(predict[0][0] * 100, 2))
            self.nc_probability.setTextVisible(True)
            self.em_probability.setValue(round(predict[0][1] * 100, 2))
            self.em_probability.setTextVisible(True)
            self.lm_probability.setValue(round(predict[0][2] * 100, 2))
            self.lm_probability.setTextVisible(True)
        elif self.model == "NC vs EM vs LM vs AD":
            model = load_model(self.fourClassModel)
            predict = model.predict(data)
            self.ad_probability.setValue(round(predict[0][0] * 100, 2))
            self.ad_probability.setTextVisible(True)
            self.nc_probability.setValue(round(predict[0][1] * 100, 2))
            self.nc_probability.setTextVisible(True)
            self.em_probability.setValue(round(predict[0][2] * 100, 2))
            self.em_probability.setTextVisible(True)
            self.lm_probability.setValue(round(predict[0][3] * 100, 2))
            self.lm_probability.setTextVisible(True)


    def center(self):
        """
        set window at the center of window
        :return:
        """
        # get the compute screen's size
        screen = QDesktopWidget().screenGeometry()
        # get the app windows' size
        size = self.geometry()
        self.move(int((screen.width() - size.width()) / 2), int((screen.height() - size.height()) / 2))

    def blank_filter(self, img):
        """
        take out blank slice
        :return:
        """
        z_min = 0
        z_max = len(img[0][0])-1
        for i in range(len(img[0][0])):
            if img[:, :, i].any() == 0:
                z_min = i
            else:
                break
        for j in range(len(img[0][0])-1, -1, -1):
            if img[:, :, j].any() == 0:
                z_max = j
            else:
                break

        x_min = 0
        x_max = len(img) - 1
        for i in range(len(img)):
            if img[i].any() == 0:
                x_min = i
            else:
                break
        for j in range(len(img)-1, -1, -1):
            if img[j].any() == 0:
                x_max = j
            else:
                break

        y_min = 0
        y_max = len(img[0])-1
        for i in range(len(img[0])):
            if img[:, i, :].any() == 0:
                y_min = i
            else:
                break
        for j in range(len(img[0])-1, -1, -1):
            if img[:, j, :].any() == 0:
                y_max = j
            else:
                break
        return z_min, z_max, x_min, x_max, y_min, y_max

    def flip90_right(self, array):
        new_arr = array.reshape(array.size)
        new_arr = new_arr[::-1]
        new_arr = new_arr.reshape(array.shape)
        new_arr = np.transpose(new_arr)[::-1]
        return new_arr

    def flip90_left(self, array):
        new_arr = np.transpose(array)
        new_arr = new_arr[::-1]
        return new_arr


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # app.setWindowIcon(QIcon("path"))
    form = MainWindow()

    form.show()
    sys.exit(app.exec_())
