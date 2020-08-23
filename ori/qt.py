import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QDesktopWidget, QPushButton, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPalette


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        self.picture1 = "/home/fan/Pictures/girl.jpg"
        self.picture2 = "/home/fan/Pictures/girl.jpg"
        self.picture3 = "/home/fan/Pictures/girl.jpg"

        super(MainWindow, self).__init__(parent)
        # self.resize(800, 500)
        self.setFixedSize(800, 500)
        self.center()
        self.status = self.statusBar()
        self.setWindowTitle("AD predictor")

        # widget = QWidget()
        self.btn1 = QPushButton(self)
        self.btn1.setText("Reset")
        self.btn1.setFixedSize(80, 40)
        self.btn1.move(440, 315)
        self.btn1.clicked.connect(self.reset_button_event)

        self.btn2 = QPushButton(self)
        self.btn2.setText("Predict")
        self.btn2.setFixedSize(80, 40)
        self.btn2.move(440, 395)
        self.btn2.clicked.connect(self.predict_button_event)

        self.btn3 = QPushButton(self)
        self.btn3.setText("Choose file")
        self.btn3.setFixedSize(80, 40)
        self.btn3.move(40, 100)

        self.label1 = QLabel(self)
        self.label1.setAutoFillBackground(True)
        # set background color
        # palette = QPalette()
        # palette.setColor(QPalette.Window, Qt.black)
        # self.label1.setPalette(palette)
        self.label1.setFixedSize(700, 50)
        self.label1.move(50, 30)
        self.label1.setAlignment(Qt.AlignCenter)
        # colors: Aquamarine,
        self.label1.setText("<font color=GreenYellow size=8>Welcome to use AD predictor</font>")

        self.label2 = QLabel(self)
        self.label2.setAutoFillBackground(True)
        self.label2.setFixedSize(240, 40)
        self.label2.move(120, 100)
        # set background color
        # palette = QPalette()
        # palette.setColor(QPalette.Window, Qt.black)
        # self.label2.setPalette(palette)
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setText("<font size=5>no file has been choosed</font>")

        self.label3 = QLabel(self)
        self.label3.setAlignment(Qt.AlignCenter)
        self.label3.setPixmap(QPixmap(self.picture1))

        self.label3.setFixedSize(256, 256)
        self.label3.move(62, 180)

        self.label4 = QLabel(self)
        self.label4.setAlignment(Qt.AlignCenter)
        self.label4.setPixmap(QPixmap(self.picture2))
        self.label4.setFixedSize(128, 128)
        self.label4.move(448, 121)

        self.label5 = QLabel(self)
        self.label5.setAlignment(Qt.AlignCenter)
        self.label5.setPixmap(QPixmap(self.picture3))
        self.label5.setFixedSize(128, 128)
        self.label5.move(624, 121)

        self.label6 = QLabel(self)
        self.label6.setText("ad <br> cn")
        self.label6.setAlignment(Qt.AlignCenter)
        self.label6.setFixedSize(128, 128)
        self.label6.move(560, 250)

    def reset_button_event(self):
        sender = self.sender()
        print(sender.text())

    def predict_button_event(self):
        sender = self.sender()
        print(sender.text())
        # sender.setText("clicked")

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # app.setWindowIcon(QIcon("path"))
    form = MainWindow()

    form.show()
    sys.exit(app.exec_())
