import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import distance
from numpy.linalg import inv
import math

from PyQt5 import QtCore,QtGui,QtWidgets, uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication,QDialog,QMainWindow,QFileDialog

#import các lớp hỗ trợ
from matrix_utils import MatrixUtils

class WindowScreen(QtWidgets.QMainWindow):
    def __init__(self):
        super(WindowScreen,self).__init__()
        uic.loadUi('datn-gui/test_gui.ui',self)
        self.setWindowTitle('Hello')
        self.pushButton.clicked.connect(self.on_pushButton_click)
        self.pushButton_2.clicked.connect(self.on_pushButton_click_2)
    @pyqtSlot()
    def on_pushButton_click(self):
        #self.
        print("Clicked")
    @pyqtSlot()
    def on_pushButton_click_2(self):
        #self.
        self.getfiles()
        print("Clicked 2")

    def getfiles(self):
        dlg = QFileDialog.Option()
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=dlg)
        if fileName : 
            print(fileName)

app = QApplication(sys.argv)
widget = WindowScreen()

widget.show()

sys.exit(app.exec_())

#Defind static value
DEFAULT_SIZE = [100, 100]
DEFAULT_IMAGE_DIR = './data/img_train_10/'


# X la ma tran du lieu
#[X, y] = MatrixUtils.read_images(DEFAULT_IMAGE_DIR,DEFAULT_SIZE)   
#  Ảnh trung bình của tập dữ liệu huấn luyện
#average_weight_matrix = np.reshape(MatrixUtils.get_data_matrix(X).mean( axis =0), X[0].shape)
