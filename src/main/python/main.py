import sys
import matplotlib
matplotlib.use('Qt5Agg')

from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QAction, QTabWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFileDialog, QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView, QLabel,  QGraphicsScene, QGraphicsView, QComboBox, QRadioButton, QLineEdit, QFormLayout, QListWidget, QPushButton, QErrorMessage, QMessageBox, QApplication, QAction, QAbstractItemView, QDialog, QCheckBox
from PyQt5.QtGui import QIcon, QPixmap, QIntValidator, QCursor, QClipboard, QImage, QPainter, QTransform, QFont, QKeySequence
from PyQt5.QtCore import pyqtSlot, QThread, pyqtSignal, QRect, QObject, Qt, QVariant, QMutex, QEvent

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
		psetAction = self.makeMenuItem("&Set p^-1", "", "&Set p^-1 File", self.setPFile)

		#reloadAction is also super unstable right now
		reloadAction = self.makeMenuItem("&Reload Plots", "CTRL+R", "Reload plots with new data", self.reload)
		copyAction = self.makeMenuItem("&Copy Plot", "", "Copy current plot to clipboard", self.clipPlot)

		self.statusBar()

		mainMenu = self.menuBar()
		fileMenu = mainMenu.addMenu('&File')
		fileMenu.addAction(saveAction)
		fileMenu.addAction(openAction)
		fileMenu.addAction(opsetAction)
		fileMenu.addAction(corrsetAction)
		fileMenu.addAction(psetAction)

		plotsMenu = mainMenu.addMenu('&Plots')
		plotsMenu.addAction(reloadAction)
		plotsMenu.addAction(copyAction)

		screen = app.screens()[0]
		dpi = screen.physicalDotsPerInch()
		self.dpi = dpi

		self.appctxt = ApplicationContext()

		fileName = self.openFileNameDialog("Select Isodat Output for Analysis", "Excel Files (*.xls *.xlsx)")

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

			#-------------------------------------------
			#preload initial dataframe to get a look
			self.ip.preload()
			#show setup diag
			self.setup = PreprocessPopup(self.ip)
			self.setup.exec()
			#wait for complete: will need to check for 'exit case' if no data is returned.
			if len(self.ip.include) == 0:
				sys.exit()
			#continue

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
		self.cori = StorageFile("../resources/standard.txt", "Corrections", self)
		self.cori.load_new_data()
		return


	@pyqtSlot()
	def setPFile(self):
		self.pp = StorageFile("../resources/p.txt", "p^-1", self)
		self.pp.load_new_data()
		return

	@pyqtSlot()
	def reload(self):
		with wait_cursor():
			self.ip.generate_all(600, self.dpi/4)
			self.table_widget.reload()
			self.ip.export()
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
		mutex.unlock()

class StorageFile():
	def __init__(self, rfp, dname, parent):
		self.appctxt = ApplicationContext()

		self.dname = dname
		self.path = self.appctxt.get_resource(rfp)
		self.parent = parent

	def get_data(self):
		f = open(self.path, "r+")
		txt = f.read()

		if(len(txt) == 0): #if file is bad...
			self.__loader(f)
		else:
			self.excel_path = txt

		tries = 2
		for i in range(tries):
			try:
				data = pd.read_excel(self.excel_path)
			except (FileNotFoundError, IsADirectoryError):
				self.__loader()
				continue
			else:
				return data

	def load_new_data(self):
		self.__loader()

	def __loader(self, fileStream=None):
		if fileStream is None:
			fileStream=open(self.path, 'w')
		fileName = self.openFileNameDialog()
		if fileName:
			self.excel_path = fileName
			fileStream.write(fileName)
		else: #no file picked
			return

	def openFileNameDialog(self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getOpenFileName(self.parent,"Select " + self.dname + " file", "","Excel Files (*.xlsx);;Excel Files (*.xls)", options=options)
		return fileName

class MyPopup(QWidget):
	def __init__(self):
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

class SmartListWidget(QWidget):
	def __init__(self, df):
		super(QWidget, self).__init__()

		self.includeRows = []
		self.ISRows = ['nLeu', 'Nonadecane', 'Caffeine'] #default values, make these checked too

		self.layout = QVBoxLayout()
		self.tableWidget = QTableWidget()

		self.targetData = df['Component'].unique()

		self.tableWidget.setRowCount(len(self.targetData))
		self.tableWidget.setColumnCount(3)

		header = self.tableWidget.horizontalHeader() 
		header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
		header.setSectionResizeMode(1, QHeaderView.Stretch)

		self.tableWidget.setHorizontalHeaderLabels(['Include', 'IS?', 'Analyte'])

		self.tableWidget.verticalHeader().setVisible(False)

		self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)

		self.tableWidget.itemChanged.connect(self.itemChanged)

		for idx, item in enumerate(self.targetData):
			chkBoxItem = QTableWidgetItem()
			chkBoxItem.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
			chkBoxItem.setCheckState(Qt.Unchecked)

			isBoxItem = QTableWidgetItem()
			isBoxItem.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
			if str(item) in self.ISRows:
				isBoxItem.setCheckState(Qt.Checked)
			else:
				isBoxItem.setCheckState(Qt.Unchecked)

			self.tableWidget.setItem(idx,0, chkBoxItem)
			self.tableWidget.setItem(idx,1, isBoxItem)
			self.tableWidget.setItem(idx,2, QTableWidgetItem(str(item)))

		self.tableWidget.setFixedHeight(500) #OK for now. will need to eventually resize. 

		self.layout.addWidget(self.tableWidget)
		self.setLayout(self.layout)

	def itemChanged(self, item):
		lookup = self.targetData[item.row()]

		if item.column() == 1 or item.column() == 0:
			other = self.tableWidget.item(item.row(), 1-item.column())
			if other is not None:
				if other.checkState() == 2 and item.checkState() == 2:
					other.setCheckState(Qt.Unchecked)

		if item.column() == 0:
			if item.checkState() == 0 and lookup in self.includeRows:
				self.includeRows.remove(lookup)
			if item.checkState() == 2:
				self.includeRows.append(lookup)
		if item.column() == 1:
			if item.checkState() == 0 and lookup in self.ISRows:
				self.ISRows.remove(lookup)
			if item.checkState() == 2 and lookup not in self.ISRows:
				self.ISRows.append(lookup)


	def getChecked(self):
		return self.includeRows, self.ISRows

#need some way to shuffle data in...
class PreprocessPopup(QDialog):
	def __init__(self, ip):
		QDialog.__init__(self)
		self.ip = ip
		#self.setWindowModality(Qt.WindowModal)

		self.setGeometry(QRect(100, 100, 400, 600)) #x,y,w,h
		self.layout = QFormLayout(self)
		self.setLayout(self.layout)
		self.setWindowTitle("Import Settings")

		self.projName = QLineEdit()
		self.projName.setFixedWidth(250)
		tempTitle = self.ip.getFileName()
		self.projName.setText(tempTitle)
		self.layout.addRow(self.tr("&Project Name:"), self.projName)

		self.projTitle = QLineEdit()
		self.projTitle.setFixedWidth(250)
		tempTitle = self.ip.getFileName()
		self.projTitle.setText(tempTitle)
		self.layout.addRow(self.tr("&Sequence Name:"), self.projTitle)

		self.SSel = QComboBox()
		self.SSel.setFixedSize(325, 50)
		for value in self.ip.df['Identifier 1'].unique():
			self.SSel.addItem(value)
		self.layout.addRow(self.tr("&AA Standard:"), self.SSel)

		self.QASel = QComboBox()
		self.QASel.setFixedSize(325, 50)
		for value in self.ip.df['Identifier 1'].unique():
			self.QASel.addItem(value)
		self.layout.addRow(self.tr("&QA Standard:"), self.QASel)

		#make checkbox list here
		self.selBox = SmartListWidget(self.ip.df)
		self.layout.addRow(self.tr("&Include Analytes:"), self.selBox)

		self.go= QPushButton('Load Data', self)
		self.go.clicked.connect(self.contData)
		self.layout.addWidget(self.go)

	@pyqtSlot()
	def contData(self):
		pname = self.projName.text()
		self.ip.projName = pname

		title = self.projTitle.text()
		self.ip.fileName = title

		opt = self.SSel.currentText()
		self.ip.sName = opt

		qaopt = self.QASel.currentText()
		self.ip.qaName = qaopt

		incl, is_c = self.selBox.getChecked()
		self.ip.include = incl
		self.ip.is_c = is_c
	
		self.close()


class PlotWebRender(QMainWindow):
	def __init__(self, fig):
		self.qapp = QApplication([])

		self.sf = 1.0

		QMainWindow.__init__(self)
		self.widget = QWidget()
		self.setCentralWidget(self.widget)
		self.widget.setLayout(QVBoxLayout())
		self.widget.layout().setContentsMargins(0,0,0,0)
		self.widget.layout().setSpacing(0)

		self.ctrlLayout = QHBoxLayout()

		self.zin = QPushButton('Zoom In', self)
		self.zin.clicked.connect(self.zoomin)
		self.ctrlLayout.addWidget(self.zin)

		self.zout = QPushButton('Zoom Out', self)
		self.zout.clicked.connect(self.zoomout)
		self.ctrlLayout.addWidget(self.zout)

		self.reset = QPushButton('Reset Zoom', self)
		self.reset.clicked.connect(self.resetz)
		self.ctrlLayout.addWidget(self.reset)

		ctrl = QWidget();
		ctrl.setLayout(self.ctrlLayout);
		ctrl.setFixedWidth(500)
		self.widget.layout().addWidget(ctrl)

		self.fig = fig
		buf = io.BytesIO()

		fig.savefig(buf, format='svg')
		buf.seek(0)
		image_data = buf.read()

		#set paint here
		render = QSvgRenderer(image_data)

		img = QPixmap(render.defaultSize())
		img.fill()
		paint = QPainter(img)
		render.render(paint)
		paint.end()

		scene = QGraphicsScene()
		scene.addPixmap(img)

		self.view = QGraphicsView(scene)
		self.widget.layout().addWidget(self.view)

		buf.close()
	"""
	def update(self, fig):
		self.fig = fig
		buf = io.BytesIO()

		fig.savefig(buf, format='svg')
		buf.seek(0)
		image_data = buf.read()

		self.svgWidget.renderer().load(image_data)
		buf.close()
	"""
	@pyqtSlot()
	def zoomin(self):
		self.sf += 0.1
		self.scale()
		pass

	@pyqtSlot()
	def zoomout(self):
		self.sf -= 0.1
		self.scale()
		pass

	@pyqtSlot()
	def resetz(self):
		self.sf = 1.0
		self.scale()
		pass

	def scale(self):
		transform = QTransform()
		transform.scale(self.sf, self.sf)

		self.view.setTransform(transform, False)

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

class NormalizeWidget(QWidget):
	def __init__(self, parent):
		super(QWidget, self).__init__(parent)
		self.mainLayout = QHBoxLayout()

		self.layout = QFormLayout()

		self.appctxt = ApplicationContext()

		#replace
		self.cori = StorageFile("../resources/standard.txt", "Corrections", self)
		
		e = self.loadData()
		if e is False: return

		self.ip = parent.ip

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

		self.go= QPushButton('Run', self)
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
		#with wait_cursor():
		e = self.loadData()
		if e is False: return

		self.corrector.set_ip(self.ip)

		mode = self.comboQC.currentData()
		standard = self.comboStandard.currentText()
		self.corrector.correct_individual(standard, mode, [])

		self.ip.export2()

		return
		"""
		#set isoplot object
		self.corrector.set_ip(self.ip)
		
		standard = self.comboStandard.currentText()
		exclusions = []
		
		if self.single.isChecked():
			mode = self.comboQC.currentData()
			self.corrector.correct_individual(standard, mode, exclusions)

		if not self.single.isChecked():
			mode = self.comboQC.currentData()
			self.corrector.correct_all(standard, mode, exclusions)

		self.ip.export()

		return
		"""

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
		data = self.cori.get_data()
		if data is None: return False

		#basic check for malformed data
		if not set(['Compound','Standard', 'd13C', 'd15N']).issubset(data.columns):
			self.handleError("Connections file is malformed. Repair and try again")
			return False

		pdata = StorageFile("../resources/p.txt", "p^-1", self).get_data()
		if pdata is None: return False

		self.corrector = Corrections(data, pdata)

		return True

class CorrectWidget(QWidget):
	def __init__(self, parent):
		super(QWidget, self).__init__(parent)
		self.mainLayout = QVBoxLayout()

		self.ip = parent.ip

		self.formData = []
		for idx, analyte in enumerate(self.ip.include):
			form = QFormLayout()

			title = QLabel()
			#
			myFont = QFont()
			myFont.setBold(True)
			title.setFont(myFont)
			#
			title.setText(analyte)
			form.addRow(title)
			
			check = QCheckBox()
			check.stateChanged.connect(self.enableIS)
			form.addRow(self.tr("&IS Correction:"), check)

			comboQC = QComboBox()
			comboQC.setFixedSize(325, 50)
			for compound in self.ip.is_c:
				comboQC.addItem(compound)
			comboQC.setEnabled(False)
			form.addRow(self.tr("&IS:"), comboQC)

			runslope = QCheckBox()
			form.addRow(self.tr("&Slope Correction:"), runslope)

			self.formData.append({'analyte': analyte, 'check': check, 'qc': comboQC, 'slope': runslope})

			wrapper = QWidget();
			wrapper.setLayout(form);
			self.mainLayout.addWidget(wrapper)

		self.scrollwrap = QWidget()
		self.scrollwrap.setLayout(self.mainLayout)

		self.scroll = QScrollArea()
		self.scroll.setWidgetResizable(True)
		self.scroll.setWidget(self.scrollwrap)

		self.finalWrap = QVBoxLayout()
		self.finalWrap.addWidget(self.scroll)

		self.go= QPushButton('Run', self)
		self.go.clicked.connect(self.runAll)
		self.go.setFixedWidth(100)
		self.finalWrap.addWidget(self.go)

		self.setLayout(self.finalWrap)

	@pyqtSlot()
	def enableIS(self):
		for item in self.formData:
			state = item['check'].checkState()
			if state == 2:
				item['qc'].setEnabled(True)
			if state == 0:
				item['qc'].setEnabled(False)

	@pyqtSlot()
	def runAll(self):
		def digest(x):
			out = {
				'compound': x['analyte'],
				'is': x['check'].checkState() == 2, 
				'is_comp': x['qc'].currentText(),
				'ro': x['slope'].checkState() == 2,
				}
			return out

		data = [digest(x) for x in self.formData]
		self.ip.aa_correct(data)
		return 

class BuildTable(QTableWidget):
	def __init__(self, cols, rows, width):
		super(QTableWidget, self).__init__()

		self.setRowCount(len(rows))
		self.setColumnCount(len(cols))

		header = self.horizontalHeader()       
		header.setSectionResizeMode(0, QHeaderView.Stretch)
		for i in range(1, len(cols)):
			header.setSectionResizeMode(i, QHeaderView.ResizeToContents)

		self.setHorizontalHeaderLabels(cols)

		for i, row in enumerate(rows):
			for j, item in enumerate(row):
				self.setItem(i,j, QTableWidgetItem(str(item)))

		self.setFixedWidth(width)

		self.verticalHeader().setVisible(False)
		self.setEditTriggers(QAbstractItemView.NoEditTriggers)

		self.clip = QApplication.clipboard()

	def keyPressEvent(self, e):
		if (e.modifiers() & Qt.ControlModifier):
			selected = self.selectedRanges()

			if e.key() == Qt.Key_C: #copy
				s = '\t'+"\t".join([str(self.horizontalHeaderItem(i).text()) for i in range(selected[0].leftColumn(), selected[0].rightColumn()+1)])
				s = s + '\n'

				for r in range(selected[0].topRow(), selected[0].bottomRow()+1):
					#s += self.verticalHeaderItem(r).text() + '\t'
					for c in range(selected[0].leftColumn(), selected[0].rightColumn()+1):
						try:
							s += str(self.item(r,c).text()) + "\t"
						except AttributeError:
							s += "\t"
					s = s[:-1] + "\n" #eliminate last '\t'
				self.clip.setText(s)      


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
		self.tab7 = QWidget()
		self.tabs.resize(300,200)

		# Add tabs
		self.tabs.addTab(self.tab1, "Overview")
		self.tabs.addTab(self.tab3, "Standard/Sample IS")
		self.tabs.addTab(self.tab4, "AA Standard")
		self.tabs.addTab(self.tab5, "Sample SD")
		self.tabs.addTab(self.tab7, "Corrections")
		self.tabs.addTab(self.tab6, "De-derivatization")

	def make_plots(self, ip):	
		std_width = 350 #todo: make parem passing betwen isoplot class and this explicit. 
		aa_width = 200
		is_width = 250

		self.overview = PlotWebRender(ip.OVER)
		
		self.aah = PlotWebRender(ip.AA_H)

		self.sdp = PlotWebRender(ip.STD_H)

		self.ish = PlotWebRender(ip.IS_H)

		self.over_tableWidget = BuildTable(['Compound', 'Slope', 'R^2'], ip.SLOPE_T, is_width)

		self.aa_tableWidget = BuildTable(['Compound', 'Row', 'd13C'], ip.AA_T, aa_width)

		#fix this
		self.tableWidget = BuildTable(['Compound', 'Sample', 'Row', 'd13C SD'], ip.STD_T, std_width)

		self.is_tableWidget = BuildTable(['Type', 'Compound', 'Row', 'd13C'], ip.IS_T, is_width)

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

	def render(self):
		#de-deriv
		self.tab6.layout = QHBoxLayout(self)
		self.tab6.layout.addWidget(NormalizeWidget(self))
		self.tab6.setLayout(self.tab6.layout)

		#corrections 2
		self.tab7.layout = QVBoxLayout(self)
		self.tab7.layout.addWidget(CorrectWidget(self))
		self.tab7.setLayout(self.tab7.layout)

		# Create tab
		self.tab1.layout = QHBoxLayout(self)
		self.tab1.layout.addWidget(self.overview)
		self.tab1.layout.addWidget(self.over_tableWidget)
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