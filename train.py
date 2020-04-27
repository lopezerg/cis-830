from __future__ import division
import os, wave, struct
import lib
import numpy as np
import ANN


if __name__ == '__main__':

	index = 0
	chunk = 256
	n = ANN.Network(14,30)
	mp = ANN.SelfOrganizingMap(n) 

	try :
		files = os.listdir("../cis 830/wav")
		for filename in files :
			try :
				wav = wave.open("../cis 830/wav/" + filename)
				bin = wav.readframes(wav.getnframes())
				data = []
				for i in range(0,len(bin)) :
					if i%2 != 0 :
						continue

					data.append(struct.unpack("<h", bin[i:i+2])[0])

				rate,chunk = (wav.getframerate(),256)
				duration = (len(data)/rate)*1000 
				featureVector = []
				for i in range(0,int(duration/10)) :
					frame = data[i*10:(i*10)+30]
					mfcc = signallib.mfcc(frame)
					output = n.update(mfcc)
					mp.update( mfcc,index,len(files) * 3 )
					index += 1

				name = filename.split('.')[0]
				print(name)

			except IOError:
				pass

	except OSError :
		pass
		
		
	n.createFile()