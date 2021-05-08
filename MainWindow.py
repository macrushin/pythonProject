from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTextEdit, QFileDialog

from DetectObjectWindow import DetectObjectWindow
from Ui_MainWindow import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButton.clicked.connect(self.showFileDialog)

    def showFileDialog(self):
        fname = QFileDialog.getOpenFileName(self.ui.centralwidget, 'Open file')[0]
        if fname:
            f = open(fname, 'r')
            self.openDetectWindow(str(f.name))
            f.close()

    def openDetectWindow(self, file_name):
        dialog = DetectObjectWindow(self, file_name)

        dialog.show()
