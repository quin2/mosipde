import sys
import matplotlib
matplotlib.use('Qt5Agg')

from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFileDialog, QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView, QLabel,  QGraphicsScene, QGraphicsView
from PyQt5.QtGui import QIcon, QPixmap, QImage, QPen, QBrush, QColor, QFont
from PyQt5.QtCore import pyqtSlot, QThread, pyqtSignal, QRect, QThreadPool, QRunnable, QObject

from PyQt5.QtSvg import QSvgWidget, QSvgRenderer, QGraphicsSvgItem

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import matplotlib.pyplot as plt

from plots import ISOplot

from pdb import set_trace as bp

import threading

import os,subprocess,time,traceback, time, io

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

#resolve missing import
import sklearn.utils._cython_blas
import sklearn.neighbors.typedefs
import sklearn.neighbors.quad_tree
import sklearn.tree
import sklearn.tree._utils

class App(QMainWindow):
	def __init__(self, app):
		super().__init__()
		self.title = 'ISOviewer'
		self.left = 10
		self.top = 30
		self.width = 1100
		self.height = 700
		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)

		screen = app.screens()[0]
		dpi = screen.physicalDotsPerInch()
		self.dpi = dpi

		fileName = self.openFileNameDialog()

		if fileName:
			#start thread
			self.t = TTT(fileName, self.dpi)
			self.t.taskFinished.connect(self.dw)
			self.t.start()

			#setup progress bar
			self.w = MyPopup()
			self.w.setGeometry(QRect(100, 100, 400, 200))
			
			#setup internal layout for future
			self.table_widget = MyTableWidget(self)
			self.setCentralWidget(self.table_widget)
			self.table_widget.build()

			#show progress bar
			self.w.show()
			self.w.startBar()

		else:
			sys.exit()

	@pyqtSlot(ISOplot)
	def dw(self, ISOplot):
		#show main window
		self.show()
		self.setGeometry(self.left, self.top, self.width, self.height)

		#close the progress bar
		self.w.stopBar()
		self.w.close()

		#display plots and render
		self.ip = ISOplot
		self.table_widget.make_plots(self.ip)

		self.table_widget.render()
		

	def openFileNameDialog(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getOpenFileName(self,"Select Isodat Output for Analysis", "","Excel Files (*.xls)", options=options)
		return fileName

	#handle window close manually
	def closeEvent(self, event):
		sys.exit()
		return

class TTT(QThread):
	taskFinished = pyqtSignal(ISOplot)

	def __init__(self, path, dpi):
		super(TTT, self).__init__()
		self.quit_flag = False
		self.path = path
		self.dpi = dpi

	def run(self):
		self.doSomething()
		self.taskFinished.emit(self.iso)
		self.quit()

	def doSomething(self):
		self.iso = ISOplot(self.path)
		self.iso.generate_all(600, self.dpi/4) #first arg is plot size in px, TODO: make dynamic. 


class MyPopup(QWidget):
    def __init__(self):
    	#init
        QWidget.__init__(self)
        self.layout = QVBoxLayout(self)
        #add text
        self.infoText = QLabel("Loading Data & Building Charts")
        self.layout.addWidget(self.infoText)

        #add progress bar
        self.progressBar = QProgressBar(self)
        self.progressBar.setRange(0,1)
        self.layout.addWidget(self.progressBar)
        
    def startBar(self):
    	self.progressBar.setRange(0,0)

    def stopBar(self):
    	self.progressBar.setRange(0,1)


class ScrollableWindow(QMainWindow):
	def __init__(self, fig):
		self.qapp = QApplication([])

		QMainWindow.__init__(self)
		self.widget = QWidget()
		self.setCentralWidget(self.widget)
		self.widget.setLayout(QVBoxLayout())
		self.widget.layout().setContentsMargins(0,0,0,0)
		self.widget.layout().setSpacing(0)

		self.fig = fig
		self.canvas = FigureCanvasQTAgg(self.fig)
		self.canvas.draw()
		self.scroll = QScrollArea(self.widget)
		self.scroll.setWidget(self.canvas)

		self.nav = NavigationToolbar(self.canvas, self.widget)
		self.widget.layout().addWidget(self.nav)
		self.widget.layout().addWidget(self.scroll)

class PlotWebRender(QMainWindow):
	def __init__(self, fig):
		self.qapp = QApplication([])

		QMainWindow.__init__(self)
		self.widget = QWidget()
		self.setCentralWidget(self.widget)
		self.widget.setLayout(QVBoxLayout())
		self.widget.layout().setContentsMargins(0,0,0,0)
		self.widget.layout().setSpacing(0)

		self.fig = fig
		buf = io.BytesIO()

		fig.savefig(buf, format='svg')
		buf.seek(0)
		image_data = buf.read()

		svgWidget = QSvgWidget()
		svgWidget.renderer().load(image_data)

		self.scroll = QScrollArea(self.widget)
		self.scroll.setWidget(svgWidget)

		self.widget.layout().addWidget(self.scroll)

		buf.close()

class MyTableWidget(QWidget):

	def __init__(self, parent):
		super(QWidget, self).__init__(parent)
		self.layout = QVBoxLayout(self)

	def build(self):
		# Initialize tab screen
		self.tabs = QTabWidget()
		self.tab1 = QWidget()
		self.tab3 = QWidget()
		self.tab4 = QWidget()
		self.tab5 = QWidget()
		self.tabs.resize(300,200)

		# Add tabs
		self.tabs.addTab(self.tab1,"Overview")
		self.tabs.addTab(self.tab3,"Test/Sample IS")
		self.tabs.addTab(self.tab4, "AA Standard")
		self.tabs.addTab(self.tab5,"SD")

		self.tw = self.tab1.frameGeometry().width()

	def buildTable(self, cols, rows, width):
		tableWidget = QTableWidget()

		tableWidget.setRowCount(len(rows))
		tableWidget.setColumnCount(len(cols))

		header = tableWidget.horizontalHeader()       
		header.setSectionResizeMode(0, QHeaderView.Stretch)
		for i in range(1, len(cols)):
			header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

		tableWidget.setHorizontalHeaderLabels(cols)

		for i, row in enumerate(rows):
			for j, item in enumerate(row):
				tableWidget.setItem(i,j, QTableWidgetItem(str(item)))

		tableWidget.setFixedWidth(width)
		return tableWidget

	def make_plots(self, ip):	
		std_width = 230 #todo: make parem passing betwen isoplot class and this explicit. 
		aa_width = 200
		is_width = 250

		self.overview = PlotWebRender(ip.OVER)
		
		self.aah = PlotWebRender(ip.AA_H)

		self.sdp = PlotWebRender(ip.STD_H)

		self.ish = PlotWebRender(ip.IS_H)

		self.aa_tableWidget = self.buildTable(['Compound', 'Row', 'd13C'], ip.AA_T, aa_width)

		self.tableWidget = self.buildTable(['Compound', 'Sample', 'd13C SD'], ip.STD_T, std_width)

		self.is_tableWidget = self.buildTable(['Type', 'Compound', 'Row', 'd13C'], ip.IS_T, is_width)

	def render(self):
		# Create tab
		self.tab1.layout = QVBoxLayout(self)
		self.tab1.layout.addWidget(self.overview)
		#layout
		self.tab1.setLayout(self.tab1.layout)

		# Create tab
		self.tab3.layout = QHBoxLayout(self)
		self.tab3.layout.addWidget(self.ish)
		self.tab3.layout.addWidget(self.is_tableWidget)
		#layout
		self.tab3.setLayout(self.tab3.layout)

		# Create tab
		self.tab4.layout = QHBoxLayout(self)
		self.tab4.layout.addWidget(self.aah)
		self.tab4.layout.addWidget(self.aa_tableWidget)
		#layout
		self.tab4.setLayout(self.tab4.layout)

		# Create tab
		self.tab5.layout = QHBoxLayout(self)
		self.tab5.layout.addWidget(self.sdp)
		self.tab5.layout.addWidget(self.tableWidget)
		#layout
		self.tab5.setLayout(self.tab5.layout)

		# Add tabs to widget
		self.layout.addWidget(self.tabs)
		self.setLayout(self.layout)

if __name__ == '__main__':
	app=QApplication(sys.argv)
	window=App(app)
	#window.show() #we're handling this in the main window so we can only show everything on loading complete
	app.exec_()