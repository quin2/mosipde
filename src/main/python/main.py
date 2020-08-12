import sys
import matplotlib
matplotlib.use('Qt5Agg')

from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFileDialog, QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView, QLabel,  QGraphicsScene, QGraphicsView, QComboBox, QRadioButton, QLineEdit, QFormLayout, QListWidget, QPushButton, QErrorMessage, QMessageBox, QApplication, QAction
from PyQt5.QtGui import QIcon, QPixmap, QIntValidator, QCursor, QClipboard, QImage
from PyQt5.QtCore import pyqtSlot, QThread, pyqtSignal, QRect, QObject, Qt, QVariant, QMutex

from PyQt5.QtSvg import QSvgWidget, QSvgRenderer, QGraphicsSvgItem

from matplotlib.figure import Figure

import matplotlib.pyplot as plt

import pandas as pd

from plots import ISOplot, Corrections

from pdb import set_trace as bp

import threading

import os,subprocess,time,traceback, time, io

from contextlib import contextmanager

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

warnings.filterwarnings("ignore",category=FutureWarning)

#resolve missing import
import sklearn.utils._cython_blas
import sklearn.neighbors.typedefs
import sklearn.neighbors.quad_tree
import sklearn.tree
import sklearn.tree._utils

class App(QMainWindow):
	def __init__(self):
		super().__init__()
		self.title = 'ISOviewer'
		self.left = 10
		self.top = 30
		self.width = 1100
		self.height = 700
		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)

		#save
		saveAction = self.makeMenuItem("&Save", "Ctrl+S", "Save Changes", self.save)
		openAction = self.makeMenuItem("&Open", "Ctrl+O", "Open ISOdat", self.on)
		openAction.setDisabled(True) #set disabled until we can make this stable...
		opsetAction = self.makeMenuItem("&Set Ouput", "", "Set Output Folder", self.setSaveFolder)
		corrsetAction = self.makeMenuItem("&Set Corrections", "", "Set Corrections File", self.setCorrFile)

		#reloadAction is also super unstable right now
		reloadAction = self.makeMenuItem("&Reload Plots", "CTRL+R", "Reload plots with new data", self.reload)
		copyAction = self.makeMenuItem("&Copy Plot", "CTRL+C", "Copy current plot to clipboard", self.clipPlot)

		self.statusBar()

		mainMenu = self.menuBar()
		fileMenu = mainMenu.addMenu('&File')
		fileMenu.addAction(saveAction)
		fileMenu.addAction(openAction)
		fileMenu.addAction(opsetAction)
		fileMenu.addAction(corrsetAction)

		plotsMenu = mainMenu.addMenu('&Plots')
		plotsMenu.addAction(reloadAction)
		plotsMenu.addAction(copyAction)

		screen = app.screens()[0]
		dpi = screen.physicalDotsPerInch()
		self.dpi = dpi

		self.appctxt = ApplicationContext()

		fileName = self.openFileNameDialog("Select Isodat Output for Analysis", "Excel Files (*.xls)")

		if fileName:
			self.ip = ISOplot(fileName)

			#get folder here
			self.opf = self.appctxt.get_resource("../resources/outputs.txt")

			#pick folder for outputs (save in memory)
			f = open(self.opf, "r+")
			txt = f.read()

			#run if we don't have a good file
			if(len(txt) == 0):
				fileName = self.saveFolderDialog()
				if fileName:
					self.ip.out_folder_path = fileName
					f.write(fileName) #save off, usually only runs on first run
				else:
					return
			else:
				self.ip.out_folder_path = txt

			#TODO:if that folder isn't around anymore, pick a new one
			if not os.path.isdir(self.ip.out_folder_path):
				fileName = self.saveFolderDialog()
				if fileName:
					self.ip.out_folder_path = fileName
					wr = open(self.opf, 'w')
					wr.write(fileName) #overwrite last
				else:
					return

			#start thread
			self.t = TTT(self.ip)
			self.t.taskFinished.connect(self.dw)
			self.t.start()

			#setup progress bar
			self.w = MyPopup()
			self.w.setGeometry(QRect(100, 100, 400, 100))
			
			#setup internal layout for future
			self.table_widget = MyTableWidget(self)
			self.setCentralWidget(self.table_widget)
			self.table_widget.build()

			#show progress bar
			self.w.show()
			self.w.startBar()

		else:
			sys.exit()

	@pyqtSlot()
	def dw(self):
		#show main window
		self.ip.generate_all(600, self.dpi/4) #first arg is plot size in px, TODO: make dynamic. 

		self.show()
		self.setGeometry(self.left, self.top, self.width, self.height)

		#close the progress bar
		self.w.stopBar()
		self.w.close()

		#display plots and render
		self.table_widget.make_plots(self.ip)

		self.t.quit()

		self.table_widget.render()
		

	def openFileNameDialog(self, header, include):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getOpenFileName(self,header, "",include, options=options)
		return fileName

	def saveFolderDialog(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog

		fileName= QFileDialog.getExistingDirectory(self, "Select location to save corrected files", options=options)
		return fileName

	def makeMenuItem(self, text, shortcut, tooltip, callback):
		menuItem = QAction(text, self)
		menuItem.setShortcut(shortcut)
		menuItem.setStatusTip(tooltip)
		menuItem.triggered.connect(callback)
		return menuItem

	@pyqtSlot()
	def save(self):
		self.ip.export()
		return

	@pyqtSlot()
	def on(self):
		self.__init__()
		return

	@pyqtSlot()
	def setSaveFolder(self):
		folderName = self.saveFolderDialog()
		if folderName:
			self.ip.out_folder_path = folderName

			wr = open(self.opf, 'w')
			wr.write(folderName) #save off
		else:
			return
		return

	@pyqtSlot()
	def setCorrFile(self):
		self.fp = self.appctxt.get_resource("../resources/standard.txt")
		fileName = self.openFileNameDialog("Select Corrections File", "Excel Files (*.xlsx)")
		if fileName:
			self.ip.corr_file_Path = fileName

			wr = open(self.fp, 'w')
			wr.write(fileName)
		else: #no file picked
			return
		return

	@pyqtSlot()
	def reload(self):
		self.ip.generate_all(600, self.dpi/4)
		#self.table_widget.make_plots(self.ip) #this is where the segfault occurs
		self.table_widget.reload()
		return

	@pyqtSlot()
	def clipPlot(self):
		self.table_widget.copy_plot()
		return

	#handle window close manually
	def closeEvent(self, event):
		sys.exit()
		return

mutex = QMutex()

class TTT(QThread):
	taskFinished = pyqtSignal()

	def __init__(self, ip):
		super(TTT, self).__init__()
		self.quit_flag = False
		self.ip = ip

	def run(self):
		self.doSomething(self.ip)
		self.taskFinished.emit()
		self.quit()

	def doSomething(self, ip):
		mutex.lock()
		ip.load()
		#ip.generate_all(600, 25) #first arg is plot size in px, TODO: make dynamic. 
		mutex.unlock()

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

		self.svgWidget = QSvgWidget()
		self.svgWidget.renderer().load(image_data)

		self.scroll = QScrollArea(self.widget)
		self.scroll.setWidget(self.svgWidget)

		self.widget.layout().addWidget(self.scroll)

		buf.close()
	def update(self, fig):
		self.fig = fig
		buf = io.BytesIO()

		fig.savefig(buf, format='svg')
		buf.seek(0)
		image_data = buf.read()

		self.svgWidget.renderer().load(image_data)
		buf.close()
	def saveImg(self):
		buf = io.BytesIO()
		self.fig.savefig(buf, format='png')
		buf.seek(0)
		image_data = buf.read()

		img = QImage()
		img.loadFromData(image_data)

		QApplication.clipboard().setImage(img, mode=QClipboard.Clipboard)
		
		buf.close()
		return

class EditableListWidget(QWidget):
	def __init__(self, parent):
		super(QWidget, self).__init__(parent)

		self.layout = QVBoxLayout()

		self.title = QLabel("Rows")
		self.layout.addWidget(self.title)

		self.listWidget = QListWidget()
		self.listWidget.itemClicked.connect(self.enableRemove)
		self.layout.addWidget(self.listWidget)

		self.buttonRow = QHBoxLayout()

		self.plus = QPushButton('Add', self)
		self.plus.clicked.connect(self.handleAdd) 
		self.buttonRow.addWidget(self.plus)

		self.minus = QPushButton('Remove', self)
		self.minus.clicked.connect(self.handleRemove) 
		self.minus.setEnabled(False)
		self.buttonRow.addWidget(self.minus)

		self.buttonContainer = QWidget()
		self.buttonContainer.setLayout(self.buttonRow)
		self.layout.addWidget(self.buttonContainer)

		self.setLayout(self.layout)

	def handleAdd(self):
		self.listWidget.addItem("0")

	def enableRemove(self, item):
		self.minus.setEnabled(True)

		if item is not None:
			item.setFlags(item.flags() | Qt.ItemIsEditable)
			item.setSelected(True)

	def handleRemove(self):
		item = self.listWidget.currentItem()
		#self.listWidget.takeItem(self.listWidget.row(item))
		#print(self.listWidget.row(item))
		index = self.listWidget.row(item)
		if index is not None:
			self.listWidget.takeItem(index)

		if self.listWidget.count() == 0:
			self.minus.setEnabled(False)

	def allItems(self):
		return [self.listWidget.item(i).text() for i in range(self.listWidget.count())]

class SmartListWidget(QWidget):
	def __init__(self, parent):
		super(QWidget, self).__init__(parent)

		self.excludeRows = []

		self.ip = parent.ip

		self.layout = QVBoxLayout()

		self.tableWidget = QTableWidget()

		self.targetData = self.ip._all[self.ip._all['Identifier 1'] == 'AA std']

		self.tableWidget.setRowCount(len(self.targetData))
		self.tableWidget.setColumnCount(4)

		header = self.tableWidget.horizontalHeader() 
		header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
		header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
		header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
		header.setSectionResizeMode(3, QHeaderView.Stretch)

		self.tableWidget.setHorizontalHeaderLabels(['Exclude', 'Row', 'ID', 'd13c'])

		self.tableWidget.verticalHeader().setVisible(False)

		self.tableWidget.itemChanged.connect(self.itemChanged)

		idx = 0
		for i, item in self.targetData.iterrows():
			chkBoxItem = QTableWidgetItem()
			chkBoxItem.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
			chkBoxItem.setCheckState(Qt.Unchecked)

			if len(item) != 0:
				self.tableWidget.setItem(idx,0, chkBoxItem)
				self.tableWidget.setItem(idx,1, QTableWidgetItem(str(i)))
				self.tableWidget.setItem(idx,2, QTableWidgetItem(item['Identifier 1']))
				self.tableWidget.setItem(idx,3, QTableWidgetItem(str(item['Ala'])))
				idx = idx + 1

		self.tableWidget.setFixedWidth(300)

		self.layout.addWidget(self.tableWidget)
		self.setLayout(self.layout)

	def itemChanged(self, item):
		lookup = self.targetData.iloc[item.row()].name
		if item.column() == 0 and item.checkState() == 0 and lookup in self.excludeRows:
			self.excludeRows.remove(lookup)

		if item.column() == 0 and item.checkState() == 2:
			self.excludeRows.append(lookup)



class NormalizeWidget(QWidget):
	def __init__(self, parent):
		super(QWidget, self).__init__(parent)
		self.mainLayout = QHBoxLayout()

		self.layout = QFormLayout()

		self.appctxt = ApplicationContext()

		#get file here
		self.fp = self.appctxt.get_resource("../resources/standard.txt")
		f = open(self.fp, "r+")
		txt = f.read()

		#run if we don't have a good file
		if(len(txt) == 0):
			fileName = self.openFileNameDialog()
			if fileName:
				self.file_Path = fileName
				f.write(fileName) #save off
			else:
				return
		else:
			self.file_Path = txt

		#load data & repair filepath if we need to
		self.loadData()

		self.ip = parent.ip

		self.many = QRadioButton("Sample Mean")
		self.many.setChecked(True)

		self.single= QRadioButton("Individual Injection")
		self.single.setChecked(False)
		self.single.toggled.connect(self.switchMode)

		whatSample = QHBoxLayout();
		whatSample.addWidget(self.many);
		whatSample.addWidget(self.single);

		container = QWidget();
		container.setLayout(whatSample);
		self.layout.addRow(self.tr("&Target:"), container)

		self.comboStandard = QComboBox()
		self.comboStandard.setFixedSize(325, 50)
		standards = self.corrector.get_all_standards()
		for standard in standards:
			self.comboStandard.addItem(standard)
		self.layout.addRow(self.tr("&QC Standard:"), self.comboStandard)

		self.comboQC = QComboBox()
		self.comboQC.setFixedSize(325, 50)
		self.comboQC.addItem("Mean of all QC", QVariant(0))
		self.comboQC.addItem("STD preceding sample measurement", QVariant(1))
		self.comboQC.addItem("STD following sample measurement", QVariant(2)) #this one is 'sticky'
		self.comboQC.addItem("Avg. of preceding/following STD", QVariant(3))
		self.layout.addRow(self.tr("&Reference Values:"), self.comboQC)

		self.exclude = EditableListWidget(self)
		self.exclude = SmartListWidget(parent)
		self.layout.addRow(self.tr("&QC Standard Exclusions:"), self.exclude)

		self.go= QPushButton('Correct', self)
		self.go.clicked.connect(self.runCorrections)
		self.layout.addWidget(self.go)

		self.formContain = QWidget();
		self.formContain.setLayout(self.layout);

		self.mainLayout.addWidget(self.formContain)

		#knit table
		#self.QChart = PlotWebRender(self.corrector.chart)
		#self.mainLayout.addWidget(self.QCchart)

		self.setLayout(self.mainLayout)

	#nothing wired up right now 
	@pyqtSlot(bool)
	def switchMode(self, state):
		return

	@pyqtSlot()
	def runCorrections(self):
		#load data again
		self.loadData()

		#set isoplot object
		self.corrector.set_ip(self.ip)
		
		standard = self.comboStandard.currentText()
		exclusions = self.exclude.excludeRows
		
		if self.single.isChecked():
			mode = self.comboQC.currentData()
			self.corrector.correct_individual(standard, mode, exclusions)

		if not self.single.isChecked():
			mode = self.comboQC.currentData()
			self.corrector.correct_all(standard, mode, exclusions)


		#save result
		self.ip.export()

		return

	def handleError(self, err):
		msg = QMessageBox()
		msg.setIcon(QMessageBox.Critical)
		msg.setText("Error")
		msg.setInformativeText(err)
		msg.setWindowTitle("Error")
		msg.exec()

	def openFileNameDialog(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getOpenFileName(self,"Select Corrections File", "","Excel Files (*.xlsx)", options=options)
		return fileName

	def saveFolderDialog(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog

		fileName= QFileDialog.getExistingDirectory(self, "Select location to save corrected files", options=options)
		return fileName

	def loadData(self):
		#check for/repair bad filepath
		tries = 3
		for i in range(tries):
			try:
				data = pd.read_excel(self.file_Path)
			except (FileNotFoundError, IsADirectoryError):
				# pick file
				fileName = self.openFileNameDialog()
				if fileName:
					self.file_Path = fileName

					wr = open(self.fp, 'w')
					wr.write(fileName)
					continue #retry after new file is picked
				else: #no file picked, end
					return
			else:
				self.corrector = Corrections(data)

		#basic check for malformed data
		if not set(['Compound','Standard', 'd13C', 'd15N']).issubset(data.columns):
			self.handleError("Connections file is malformed. Repair and try again")
			return

		return

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
		self.tab6 = QWidget()
		self.tabs.resize(300,200)

		# Add tabs
		self.tabs.addTab(self.tab1, "Overview")
		self.tabs.addTab(self.tab3, "Test/Sample IS")
		self.tabs.addTab(self.tab4, "AA Standard")
		self.tabs.addTab(self.tab5, "SD")
		self.tabs.addTab(self.tab6, "De-derivatization")

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

		#i think this is causing it
		tableWidget.cellChanged.connect(self.tableEdited) #connect to 
		return tableWidget

	@pyqtSlot(int, int)
	def tableEdited(self, r, c):

		if(self.tabs.currentIndex() == 1 or self.tabs.currentIndex() == 2):
			#only change for first two tabs for now...
			if(self.tabs.currentIndex() == 1):
				#get old item
				comp = self.is_tableWidget.item(r, 1).text()
				row = int(self.is_tableWidget.item(r, 2).text())
				#get new item
				new = float(self.is_tableWidget.item(r, c).text())

			elif(self.tabs.currentIndex() == 2):
				#get old item
				comp = self.aa_tableWidget.item(r, 0).text()
				row = int(self.aa_tableWidget.item(r, 1).text())
				#get new item
				new = float(self.aa_tableWidget.item(r, c).text())

			self.ip.changeCell(row, comp, new)
		return

	def make_plots(self, ip):	
		std_width = 230 #todo: make parem passing betwen isoplot class and this explicit. 
		aa_width = 200
		is_width = 250

		self.overview = PlotWebRender(ip.OVER)
		
		self.aah = PlotWebRender(ip.AA_H)

		self.sdp = PlotWebRender(ip.STD_H)

		self.ish = PlotWebRender(ip.IS_H)

		#segfaulting past here...
		self.aa_tableWidget = self.buildTable(['Compound', 'Row', 'd13C'], ip.AA_T, aa_width)

		self.tableWidget = self.buildTable(['Compound', 'Sample', 'd13C SD'], ip.STD_T, std_width)

		self.is_tableWidget = self.buildTable(['Type', 'Compound', 'Row', 'd13C'], ip.IS_T, is_width)

		self.ip = ip

	def copy_plot(self):
		cw = self.tabs.currentWidget()
		first = cw.layout.itemAt(0).widget()
		if isinstance(first, PlotWebRender):
			first.saveImg()
		return

	#if we change any of the views, we need to change this also. 
	def reload(self):
		def updatePlot(tab, plot):
			layout = tab.layout
			stuff = layout.itemAt(0).widget()
			stuff.update(plot)
		def updateTableWidget(tab, data):
			layout = tab.layout
			currentTable = layout.itemAt(1).widget()
			currentTable.blockSignals(True)
			for i, row in enumerate(data):
				for j, item in enumerate(row):
					currentTable.setItem(i,j, QTableWidgetItem(str(item)))
			currentTable.blockSignals(False)	

		#knit plots
		updatePlot(self.tab1, self.ip.OVER)
		updatePlot(self.tab3, self.ip.IS_H)
		updatePlot(self.tab4, self.ip.AA_H)
		updatePlot(self.tab5, self.ip.STD_H)

		#knit tables
		updateTableWidget(self.tab3, self.ip.IS_T)
		updateTableWidget(self.tab4, self.ip.STD_T)
		updateTableWidget(self.tab5, self.ip.AA_T)

		return

	def clearLayout(self,layout):
		#for i in reversed(range(layout.count())): 
		print("starting clear")
		
		print(stuff)
		

		print("done")
		return

	def render(self):
		#turning off until we can solve thread crash
		#not causing bad access though...
		#creatte normalizer
		self.tab6.layout = QHBoxLayout(self)
		self.tab6.layout.addWidget(NormalizeWidget(self))
		self.tab6.setLayout(self.tab6.layout)

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

@contextmanager
def wait_cursor():
	try:
		QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
		yield
	finally:
		QApplication.restoreOverrideCursor()

if __name__ == '__main__':
	app=QApplication(sys.argv)
	window=App()
	#window.show() #we're handling this in the main window so we can only show everything on loading complete
	app.exec_()