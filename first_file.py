#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import (QMainWindow, QTextEdit,
	QAction, QFileDialog, QApplication,QMessageBox, QLabel, QGridLayout, QWidget, QPushButton, QVBoxLayout)
from PyQt5.QtGui import QIcon, QPixmap

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
from os import listdir
import math
from sklearn import preprocessing

class Prediction:
	def __init__(self, array):
		self.array = array

	def argmax(self):
		return np.argmax(self.array)

	def getArray(self):
		return self.array

	def normalize(self):
		sum = 0
		for cur in self.array:
			sum += cur

		i = 0
		for cur in self.array:
			self.array[i] = self.array[i] / sum * 100
			i += 1
		# self.array = preprocessing.normalize(self.array)

	def output(self):
		sum = 0
		for cur in self.array:
			sum += cur

		i = 0
		a = []
		for cur in self.array: 
			a.append( str(round(self.array[i] / sum * 100,2)))
			# self.array[i] = self.array[i] / sum * 100
			i += 1
		return a

	def len(self):
		return np.size(self.array)

class Integrator:
	def __init__(self):
		self.list = []

	def add(self, pred):
		self.list.append(pred)

	def avg(self):
		predCnt = len(self.list)
		if predCnt > 0:
			classCnt = np.size(self.list[0].getArray())

			avgList = np.zeros(classCnt)
			i = 0
			for pred in self.list:

				# k = 0.2
				# if(i==0):
				#	 k = 0.8

				j = 0
				# self.list[j].normalize()
				for elem in pred.getArray()[0]:
					avgList[j] += elem #* k
					j += 1
				i += 1

			i = 0
			while i < classCnt:
				avgList[i] /= predCnt
				i += 1
			return  Prediction(avgList)

	def geomMean(self):
		predCnt = len(self.list)
		if predCnt > 0:
			classCnt = np.size(self.list[0].getArray())

		i = 0
		for pred in self.list:
			self.list[i].normalize()

		geomList = np.copy(self.list[0].getArray())[0]

		i = 0
		for elem in geomList:
			geomList[i] = math.sqrt(elem * self.list[1].getArray()[0][i])
			i += 1

		return Prediction(geomList)

	def getPrediction(self, i):
		return list[i]

class Network:
	def __init__(self, imWidth, imHeight):
		self.imWidth = imWidth
		self.imHeight = imHeight


	def load(self, path):
		self.model = load_model(path)

	def predict(self, fname):
		image = load_img(fname, target_size=(self.imWidth, self.imHeight))
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)
		image = image / 255
		pred = self.model.predict(image)
		return Prediction(pred)

	# def classCnt(self):
	#	 return self.model.o


class Evaluator:

	def __init__(self, classCnt):
		self.rating = np.zeros(classCnt)
		self.cnt = 0
		self.good = 0

	def add(self, pred):
		winnerClass = pred.argmax()
		self.rating[winnerClass] += 1

		if(winnerClass == (int)(self.cnt / 400)):
			self.good += 1

		self.cnt += 1

	def getPositive(self):
		return self.good

	def getCnt(self):
		return self.cnt

	def getAccuracy(self):
		return self.good * 1.0 / self.cnt

	def output(self):
		text = 'Images: ' + str(self.getCnt()) + '\n' + 'Positive: ' + str(self.getPositive()) + '\n' + 'Accuracy: ' + str(self.getAccuracy())
		return text


class Base:
	def __init__(self):
		self.cars = []
		f = open('base.txt', 'r')
		for line in f:
			self.cars.append(line)
		# self.cars.append('Vaz 21099')
		# self.cars.append('Renault Logan')
		# self.cars.append('Lada Priora')
		# self.cars.append('Hyundai Solaris')
		# self.cars.append('Nissan Sunny')

	def getClass(self, maxClass):
		return self.cars[maxClass]

class Main(QMainWindow):

	base = Base()
	nets = []
	# simpleModel = Network(150, 150)
	# simpleModel.load('model_for_5.h5')
	# nets.append(simpleModel)
	#
	# simpleCroppedModel = Network(150, 150)
	# simpleCroppedModel.load('new_model_for_5.h5')
	# nets.append(simpleCroppedModel)

	# inc_model = Network(299, 299)   theano????????//
	# inc_model.load('inception_model.h5')
	# nets.append(inc_model)
	#
	inc_model2 = Network(299,299)
	inc_model2.load('inception_model_110_3.h5')
	nets.append(inc_model2)

	#resnet_model = Network(224,224)
	#resnet_model.load('resnet_model_new.h5')
	#nets.append(resnet_model)

	

	curname = '1.jpg'



	def predict(self, fname):
		integrator = Integrator()
		for net in self.nets:
			pred = net.predict(fname)
			integrator.add(pred)
		# print(integrator.list[0].getArray())
		avgPred = integrator.avg()
		# avgPred = integrator.geomMean()
		return  avgPred

	def recognize(self):
		pred = self.predict(self.curname)
		predResStr = pred.output()[:]
		maxClass = pred.argmax()
		text = self.log.toPlainText() + '\n' 
		text += self.curname + '\n' + 'Recognition result: \n' + predResStr[maxClass] + '%  - ' + self.base.getClass(maxClass)
		# text += '\n ' + str(pred.output())
		self.log.setText(text)
		
	def top5(self):
		pred = self.predict(self.curname)
		predRes = pred.getArray()
		predResStr = pred.output()[:]
		text = self.log.toPlainText() + '\n' 
		
		for cur in range(5):
			maxClass = predRes.argmax()
			text += str(cur + 1) + '. ' + predResStr[maxClass] + '%  - '+ self.base.getClass(maxClass)
			predRes[maxClass] = 0
		
		self.log.setText(text)


	def __init__(self):
		super().__init__()

		self.initUI()


	def initUI(self):

		self.statusBar()

		openFile = QAction(QIcon('open.png'), 'Open image', self)
		openFile.setShortcut('Ctrl+O')
		openFile.setStatusTip('Open new File')
		openFile.triggered.connect(self.showDialog)

		openDir = QAction(QIcon('open.png'), 'Open dir', self)
		openDir.setShortcut('Ctrl+D')
		openDir.setStatusTip('Open dir')
		openDir.triggered.connect(self.showDialogDir)

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(openFile)
		fileMenu.addAction(openDir)

		# end menu bar
		self.image = QLabel(self)
		pixmap = QPixmap('1.jpg')
		self.image.setPixmap(pixmap)
		self.log = QTextEdit()


		top5Button = QPushButton('Top 5', self)
		top5Button.clicked.connect(self.top5)

		self.curname = '1.jpg'
		recognizeButton = QPushButton('Recognize', self)
		recognizeButton.clicked.connect(lambda: self.recognize())
		# recognizeButton.clicked.connect(lambda: self.predict(self.curname))
		#end buttons



		grid = QGridLayout()
		grid.setSpacing(10)

		grid.addWidget(self.image, 1, 0, 1, 2)
		grid.addWidget(top5Button,2,0)
		grid.addWidget(recognizeButton, 2, 1)
		grid.addWidget(self.log, 3, 0, 1, 2)




		mainWidget = QWidget()
		mainWidget.setLayout(grid)
		self.setCentralWidget(mainWidget)

		self.setGeometry(300, 300, 400, 350)
		self.setWindowTitle('Car recognitor')
		self.show()




		
	def isInTopN(self, pred, n, answer):
		t = []
		for x in range(len(pred)):
			t.append((x, pred[x]))
		t.sort(key=lambda x: x[1])
		t.reverse()
		
		for x in range(0, n - 1):
			if t[x][0] == answer:
				return 1
		return 0
		
		

	def showDialog(self):
		self.curname = QFileDialog.getOpenFileName(self, 'Open photo to predict', '/home')[0]

		if self.curname:
			pixmap = QPixmap(self.curname)
			self.image.setPixmap(pixmap.scaled(299,299))

	def showDialogDir(self):
		dirName = QFileDialog.getExistingDirectory(self, 'Open dir with samples', '/home', QFileDialog.ShowDirsOnly)
		n = 110	
		array = [[0] * n for x in range(n)]
		top = 5
		arrayTopN = [0] * n
		folderCnt = 0
		if dirName:
			for subdir in listdir(dirName):
				for fname in listdir(dirName + '/' + subdir):
					pred = self.predict(dirName + '/' + subdir + '/' + fname)
					array[folderCnt][pred.argmax()] += 1
					if self.isInTopN(pred.getArray(), top, folderCnt) > 0:
						arrayTopN[folderCnt]+=1
					
					#evaluator.add(pred)
					print(fname + ' ' + str(pred.argmax()) + '\n')

				folderCnt+= 1
		
		
			#text = 'Directory: ' + dirName + '\n'
			#text += evaluator.output()
			#self.log.setText(text)
			# QMessageBox.about(self, "My message box", self.stat.,'sss')
			file = "ress.txt"
			f = open(file, "w")
			for i in range(0,n):
				for j in range(0,n):
					template =" %4d," if j < n - 1 else " %4d\n"
					f.write(template % (array[i][j]))
			f.close()
			
			file = "top5.txt"
			f = open(file, "w")
			for i in range(0,n):
				template =" %4d,"
				f.write(template % (arrayTopN[i]))
			f.close()





if __name__ == '__main__':

	app = QApplication(sys.argv)
	ex = Main()
	sys.exit(app.exec_())