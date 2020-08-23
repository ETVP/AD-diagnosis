import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QFont, QImage, QColor, QPainter
from PyQt5.QtCore import Qt, QSize, QEvent, QObject
from PyQt5.QtGui import QPixmap, QPalette, QDragEnterEvent
from PyQt5 import QtGui
import numpy as np
import nibabel as nib
import pylab
import os


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        self.appName = 'AD predictor'
        self.picture = r'/home/fan/Desktop/processed/ad/Patient Output Volume.nii'
        self.img_path = r'/home/fan/Desktop/processed/ad/Patient Output Volume.nii'
        self.page_flip = False
        self.pre_point = [0, 0]

        img_data = nib.load(self.img_path).get_data()
        min_, max_ = self.blank_filter(img_data)

        pylab.axis('off')

        pylab.imshow(img_data[:, :, int((min_ + max_) / 2)], cmap=pylab.cm.bone)
        pylab.savefig("tmp.jpg")
        self.picture = "tmp.jpg"

        super(MainWindow, self).__init__(parent)

        self.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.white)
        self.setPalette(palette)

        # self.resize(800, 500)
        self.setFixedSize(800, 500)
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

        load_data_bt = QPushButton(self)
        load_data_bt.setText("Load Data")
        load_data_bt.setFixedSize(100, 30)
        load_data_bt.move(100, 190)
        load_data_bt.clicked.connect(self.load_data_event)

        predict_bt = QPushButton(self)
        predict_bt.setText("Predict")
        predict_bt.setFixedSize(100, 30)
        predict_bt.move(220, 190)

        heatmap_bt = QPushButton(self)
        heatmap_bt.setText("Heatmap")
        heatmap_bt.setFixedSize(100, 30)
        heatmap_bt.move(100, 230)

        self.slide = QSlider(Qt.Horizontal, self)
        self.slide.setMinimum(min_)
        self.slide.setMaximum(max_)
        self.slide.setValue(int((min_ + max_) / 2))
        self.slide.setSingleStep(1)
        self.slide.setAutoFillBackground(True)
        self.slide.setPalette(palette)
        self.slide.setFixedSize(370, 30)
        self.slide.setTickInterval(5)
        self.slide.setTickPosition(QSlider.TicksBelow)
        self.slide.move(410, 20)
        self.slide.valueChanged.connect(self.slide_event)

        black_palette = QPalette()
        black_palette.setColor(QPalette.Window, Qt.black)
        self.label = QLabel(self)
        self.label.setFixedSize(390, 400)
        self.label.move(400, 70)
        self.label.setAutoFillBackground(True)
        self.label.setPalette(black_palette)

        self.img_label = QLabel(self)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setPixmap(QPixmap(self.picture))
        self.img_label.setFixedSize(341, 256)
        self.img_label.setVisible(False)
        self.img_label.move(429, 147)
        # self.img_label.setAcceptDrops(True)
        self.setAcceptDrops(True)
        # self.setAcceptDrops(True)

        self.file_label = QLabel(self)
        self.file_label.setAlignment(Qt.AlignLeft)
        self.file_label.setFixedSize(100, 30)
        self.file_label.setText("<font color=white>Choose file</font>")
        self.file_label.move(410, 430)

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.move(40, 270)
        # self.result_label.setText("Hello")
        palette = QPalette()
        # palette.setColor(QPalette.Window, Qt.gray)
        # self.result_label.setBackgroundRole(Qt.gray)
        self.result_label.setStyleSheet("background-color:lightGrey;")
        # self.result_label.setPalette(palette)
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

        self.ad_label = QLabel(self)
        self.ad_label.setAlignment(Qt.AlignCenter)
        self.ad_label.setText(" AD ")
        self.ad_label.setFixedSize(50, 30)
        self.ad_label.move(50, 330)
        self.ad_label.setFont(font2)

        self.combo = QComboBox(self)
        self.combo.addItem("NC vs AD")
        self.combo.addItem("NC vs EM")
        self.combo.addItem("AD vs LM")
        self.combo.addItem("NC vs MCI vs AD")
        self.combo.addItem("NC vs EM vs LM vs AD")
        self.combo.setFixedSize(100, 30)
        self.combo.move(220, 230)


        # self.heatmap_label.setFixedSize()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        # self.drawRectangles(qp)
        color = QColor(200, 0, 0)
        color.setNamedColor("#d4d4d4")
        qp.setPen(color)
        qp.setBrush(QColor(200,0,0))
        # qp.drawRect(60, 330, 50, 50)
        qp.drawRect(10, 15, 90, 60)
        qp.end()

    def drawRectangles(self, qp):
        color = QColor(0, 0, 0)
        color.setNamedColor("#d4d4d4")
        qp.setPen(color)
        qp.setBrush(QColor(200, 0, 0))
        qp.drawRect(110, 330, 50, 30)


    def get_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open File', "./", "nifty image (*.nii)")
        return fname

    def dragEnterEvent(self, a0: QDragEnterEvent):
        file_path = a0.mimeData().text()
        file_path = file_path[7:]
        self.img_path = file_path.strip()
        img_data = nib.load(self.img_path).get_data()
        min_, max_ = self.blank_filter(img_data)

        self.slide.setMinimum(min_)
        self.slide.setMaximum(max_)
        self.slide.setValue(int((min_ + max_) / 2))

        pylab.axis('off')
        pylab.imshow(img_data[:, :, int((min_ + max_) / 2)], cmap=pylab.cm.bone)
        pylab.savefig("tmp.jpg")
        self.picture = "tmp.jpg"
        self.img_label.setPixmap(QPixmap(self.picture))
        self.img_label.setVisible(True)
        a0.accept()

    def dropEvent(self, evn):
        pass

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

        if not os.path.exists(self.img_path):
            return

        img_data = nib.load(self.img_path).get_data()
        min_, max_ = self.blank_filter(img_data)

        self.slide.setMinimum(min_)
        self.slide.setMaximum(max_)
        self.slide.setValue(int((min_ + max_) / 2))

        pylab.axis('off')
        pylab.imshow(img_data[:, :, int((min_ + max_) / 2)], cmap=pylab.cm.bone)
        pylab.savefig("tmp.jpg")
        self.picture = "tmp.jpg"
        self.img_label.setPixmap(QPixmap(self.picture))
        self.img_label.setVisible(True)
        self.file_label.setText("<font color=white>" + self.img_path.split("/")[-1] + "</font>")

    def slide_event(self):
        sender = self.sender()
        slice = int(sender.value())
        img_data = nib.load(self.img_path).get_data()
        pylab.axis('off')
        pylab.imshow(img_data[:, :, slice], cmap=pylab.cm.bone)
        pylab.savefig("tmp.jpg")
        self.picture = "tmp.jpg"
        self.img_label.setPixmap(QPixmap(self.picture))

    def predict_button_event(self):
        sender = self.sender()
        print(sender.text())
        # sender.setText

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
        min_ = 0
        max_ = len(img[0][0])-1
        for i in range(len(img[0][0])):
            if img[:, :, i].any() == 0:
                min_ = i
            else:
                break
        for j in range(len(img[0][0])-1, -1, -1):
            if img[:, :, j].any() == 0:
                max_ = j
            else:
                break
        return min_, max_

    def mousePressEvent(self, a0: QtGui.QMouseEvent):
        # 获得相对于屏幕的坐标
        point = a0.globalPos()
        # 获得相对于窗口的坐标
        windowPoint = self.mapFromGlobal(point)
        # 获得相对于控件的坐标
        point = self.label.mapFromGlobal(point)
        if 0 < point.x() < 390 and 0 < point.y() < 400:
            self.pre_point = [point.x(), point.y()]
            self.page_flip = True

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent):
        # 获得相对于屏幕的坐标
        point = a0.globalPos()
        # 获得相对于窗口的坐标
        windowPoint = self.mapFromGlobal(point)
        # 获得相对于控件的坐标
        point = self.label.mapFromGlobal(point)

        if self.page_flip:
            # if point.x() < point[0]:
            #     self.slide.setValue(self.slide.value() - int((point.x() - point[0]) / 20))
            #

            if point.x() <= 195:
                self.slide.setValue(self.slide.value()-int((195 - point.x()) / 50))
            else:
                self.slide.setValue(self.slide.value() + int(point.x() -195) / 50)

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent):
        self.page_flip = False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # app.setWindowIcon(QIcon("path"))
    form = MainWindow()
    form.show()
    app.installEventFilter(form)
    sys.exit(app.exec_())
