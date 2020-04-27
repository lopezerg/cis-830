from __future__ import division
from ctypes import *
import pyaudio, sys, threading, struct
from PyQt5.QtWidgets import QWidget, QCheckBox, QApplication,QPushButton,QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5 import sip
import numpy as np

from canvas import Canvas
import lib
from HMM import Hmm
import record

class Speech_Recognition(Hmm) :
	def __init__(self,path) :
		super(Speech_Recognition,self).__init__()

# The back end analytics object
class Analytics() :
	def __init__(self,parent) :
		self.parent = parent

	def handle_mic_event(self,event) :
		stream,rate,chunk = event.unpack()

		duration = (len(stream)/rate)*1000
		
		featureVector = []
		for i in range(0,int(duration/10)) :
			frame = stream[i*10:(i*10)+30]
			featureVector.append(lib.mfcc(frame))

		print(featureVector)
	

# main application window
class Application(QWidget) :
	def __init__(self) :
		super(Application, self).__init__()
		self.analytics = Analytics(self)
		self.mic = record.Microphone(self,rate=8000,chunk=256,handler=self.analytics)
		self.size = (50,100)
		self.installEventFilter(self)
		self.initUI()

	def initUI(self):
		self.setGeometry(300,300,self.size[0],self.size[1])
		self.setWindowTitle('CIS 830')
		self.recordButton = QPushButton('Record', self)
		self.recordButton.clicked.connect(self.mic.startStream)
		self.pauseButton = QPushButton('Pause', self)
		self.pauseButton.clicked.connect(self.mic.pause)

		# horizontal box
		hbox = QHBoxLayout()
		hbox.addWidget(self.recordButton)
		hbox.addWidget(self.pauseButton)
		self.setLayout(hbox)

		self.show()

	def eventFilter(self,object,event) :
		return False

if __name__ == '__main__': 

	app = QApplication(sys.argv)
	ex = Application()
	sys.exit(app.exec_())
