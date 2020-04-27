import pyaudio, sys, threading, struct

class Record_Event() :
	def __init__(self,stream,rate,chunk) :
		self.stream = stream
		self.rate = rate
		self.chunk = chunk

	def unpack(self) :
		return (self.stream,self.rate,self.chunk)


class Microphone() :

	def __init__(self,parent,rate,chunk,handler) :

		self.p = pyaudio.PyAudio()
		self.chunk = chunk # number of samples per callback
		self.format = pyaudio.paInt16
		self.channels = 1
		self.rate = rate
		self.active = False
		self.parent = parent
		self.handler = handler

	def __callback(self,in_data, frame_count, time_info, status) :

		extend = [] 
		for i in range(0,len(in_data)) :
			if i%2 != 0 :
				continue
			# convert binary into signed int
			extend.append(struct.unpack("<h", in_data[i:i+2])[0])
		self.streamData.extend(extend)

		if self.active :
			return (in_data, pyaudio.paContinue)
		return (in_data, pyaudio.paComplete)

	def startStream(self) :
		self.streamData = []
		self.active = True
		self.stream = self.p.open(
			format=self.format,
			channels=self.channels,
			rate=self.rate,
			input=True,
			frames_per_buffer=self.chunk,
			stream_callback=self.__callback )
		self.stream.start_stream()

	def pause(self) :
		if self.stream.is_active() :
			self.active = False

		event = Record_Event(
			self.streamData,
			self.rate,
			self.chunk)
		if self.handler :
			self.handler.handle_mic_event(event)
