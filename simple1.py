from PyQt5 import QtWidgets
from PyQt5.QtCore import *
import sys
import time

from plots import ISOplot


class TTT(QThread):
    taskFinished = pyqtSignal(ISOplot)

    def __init__(self, path):
        super(TTT, self).__init__()
        self.quit_flag = False
        self.path = path

    """
    def run(self):
        while True:
            if not self.quit_flag:
                self.doSomething()
                time.sleep(1)
            else:
                break

        self.quit()
        self.wait()
    """

    def run(self):
        self.doSomething()
        self.taskFinished.emit(self.iso)
        self.quit()

    def doSomething(self):
        self.iso = ISOplot(self.path)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.btn = QtWidgets.QPushButton('run process')
        self.btn.clicked.connect(self.create_process)
        self.setCentralWidget(self.btn)

    def create_process(self):
        if self.btn.text() == "run process":
            print("Started")
            self.btn.setText("stop process")

            path = "/Users/quinnvinlove/Documents/sugarsBio/excel/24Sept19.xls"

            self.t = TTT(path)
            self.t.taskFinished.connect(self.yolo)
            self.t.start()
        else:
            self.t.quit_flag = True
            print("Stop sent")
            self.t.wait()
            print("Stopped")
            self.btn.setText("run process")

    @pyqtSlot(ISOplot)
    def yolo(self, ISOplot):
        print(ISOplot.done)


if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())