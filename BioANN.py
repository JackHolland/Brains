import numpy as np
#pybrain stuff
import pybrain
from pybrain.datasets	import ClassificationDataSet

from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader

import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

import json

#Default parameters
micro_dim = 2 #the number of input nodes
num_hidden = 3 #number of hidden layers
inc_bias = False #to include bias or not

momentum = 0.1 #momentum 
weight_decay = 0.01 #weight decay

myError = 10.0


max_iterations = 450000 #maximum number of iterations
num_epochs = 5 #number of ephochs at every iteration
snapshot = 1000 #when to snapshot the ANN

def load_params():
	with open("params", 'r') as f_in:
		params = json.load(f_in)

	global micro_dim
	global num_hidden
	global inc_bias
	global momentum
	global weight_decay
	global momentum
	global max_iterations
	global num_epochs
	global snapshot
	micro_dim = int(params["micro_dim"]) 
	num_hidden = int(params["num_hidden"])
	inc_bias = bool(params["inc_bias"])

	momentum = float(params["momentum"])
	weight_decay = float(params["weight_decay"])
	max_iterations = int(params["max_iterations"])
	num_epochs = int(params["num_epochs"])
	snapshot = int(params["snapshot"])



def extract_data(filename, ds):
	with open(filename, 'r') as f:
		num = 0
		for line in f:
			#skip first line
			if num == 0:
				num = num + 1
				continue
			line = line.split(',')
			label = int(line[1])
			microarray = [float(i) for i in line[3:]]
			ds.addSample(microarray, (label))
			num = num + 1

def restart(filename, netfile):
	load_params()
	#import dataset
	ds = ClassificationDataSet(micro_dim, 1, nb_classes=2)
	extract_data(filename, ds)

	tr, val = ds.splitWithProportion(0.10) #10% validation data
	#softmax output layer
	tr._convertToOneOfMany()
	val._convertToOneOfMany()

	ann = NetworkReader.readFrom(netfile)
	#restart training TODO
	done = False
	temp = netfile.split("_")

	iteration = int(temp[0])
	trainer = BackpropTrainer(ann,	dataset=tr, momentum=momentum, verbose=True, weightdecay=weight_decay)

	while(not done):
		trainer.trainEpochs(num_epochs)
		error = percentError(trainer.testOnClassData(), tr['class'])
		print "iter %d" % iteration
		#print "Error = %f" % float(error)
		print error
		if iteration >= max_iterations:
			done = True
		#pickle every 5 iterations
		if iteration%snapshot == 0:
			print "Saving model %d_%d.xml..."%(iteration, int(error))
			NetworkWriter.writeToFile(ann, "%d_%d.xml"%(iteration, int(error)))

		iteration = iteration + 1

	#testing
	tstresult = percentError(trainer.testOnClassData(dataset=val), val['class'])
	print tstresult

def train(filename):
	load_params()
	#import dataset
	ds = ClassificationDataSet(micro_dim, 1, nb_classes=2)
	extract_data(filename, ds)

	tr, val = ds.splitWithProportion(0.10) #10% validation data
	#softmax output layer
	tr._convertToOneOfMany()
	val._convertToOneOfMany()

	#build network
	ann = buildNetwork(tr.indim, num_hidden, tr.outdim, outclass=SoftmaxLayer, bias=inc_bias)

	#training 
	trainer = BackpropTrainer(ann,	dataset=tr, momentum=momentum, verbose=True, weightdecay=weight_decay)
	done = False
	iteration = 0


	while(not done):
		trainer.trainEpochs(num_epochs)
		error = percentError(trainer.testOnClassData(), tr['class'])
		print "iter %d" % iteration
		#print "Error = %f" % float(error)

		if iteration >= max_iterations:
			done = True
		#pickle every 5 iterations
		if iteration%snapshot == 0:
			print "Saving model %d_%d.xml..."%(iteration, int(error))
			NetworkWriter.writeToFile(ann, "%d_%d.xml"%(iteration, int(error)))
			print error

		iteration = iteration + 1

	#testing
	tstresult = percentError(trainer.testOnClassData(dataset=val), val['class'])
	print tstresult

if __name__ == '__main__':
	operation = sys.argv[1]
	if operation in "train":
		train(sys.argv[2])
	else:
		restart(sys.argv[2], sys.argv[3])














