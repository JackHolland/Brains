import numpy as np
#pybrain stuff
import pybrain
from pybrain.datasets	import ClassificationDataSet

from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import sys
import cPickle


micro_dim = 2
num_hidden = 3
inc_bias = False

mom = 0.1 #momentum 
wd = 0.01 #weight decay

myError = 10.0

def extract_data(filename, ds):
	with open(filename, 'r') as f:
		for line in f:
			line = line.split(',')
			label = int(line[0])
			microarray = [float(i) for i in line[1:]]
			ds.addSample(microarray, (label))



if __name__ == '__main__':

	#import dataset
	ds = ClassificationDataSet(micro_dim, 1, nb_classes=2)
	extract_data("data.csv", ds)

	tr, val = ds.splitWithProportion(0.10) #10% test data
	#softmax output layer
	tr._convertToOneOfMany()
	val._convertToOneOfMany()

	#build network
	ann = buildNetwork(tr.indim, num_hidden, tr.outdim, outclass=SoftmaxLayer, bias=inc_bias)

	#training 
	trainer = BackpropTrainer(ann,	dataset=tr, momentum=mom, verbose=True, weightdecay=wd)
	done = False
	iteration = 0

	while(not done):
		trainer.trainEpochs(5)
		error = percentError(trainer.testOnClassData(), tr['class'])
		print "iter %d" % iteration
		#print "Error = %f" % float(error)
		print error
		if error < myError:
			done = True
		#pickle every 5 iterations
		if iteration%5 == 0:
			print "Saving model..."
			with open("%d_%d.pkl"%(iteration, int(error)), 'wb') as f:
				cPickle.dump(trainer, f)

		iteration = iteration + 1

	#testing
	tstresult = percentError(trainer.testOnClassData(dataset=val), val['class'])













