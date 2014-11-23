import numpy as np
#pybrain stuff
import pybrain
from pybrain.datasets	import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SigmoidLayer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import json
from scipy import array, where, argmax
import matplotlib.pyplot as plt
import re
from graph_tool.all import *
from math import log10

#Default parameters
micro_dim = 24 #the number of input nodes
num_classes = 3
num_hidden = [48,12] #number of hidden layers
inc_bias = True #to include bias or not

momentum = 0.1 #momentum 
weight_decay = 0.1 #weight decay

myError = 10.0


max_iterations = 1000000 #maximum number of iterations
num_epochs = 5 #number of ephochs at every iteration
snapshot = 10 #when to snapshot the ANN

shapes = [(24, 48), (48, 3)]

def load_params():
	with open('params', 'r') as f_in:
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
	micro_dim = int(params['micro_dim']) 
	num_hidden = params['num_hidden']
	inc_bias = bool(params['inc_bias'])

	momentum = float(params['momentum'])
	weight_decay = float(params['weight_decay'])
	max_iterations = int(params['max_iterations'])
	num_epochs = int(params['num_epochs'])
	snapshot = int(params['snapshot'])



def extract_data(filename, ds):
	with open(filename, 'r') as f:
		lines = f.read().splitlines(True)[1:]
		for line in lines:
			cells = line.split(',')
			label = int(cells[1])
			microarray = [float(i) for i in cells[3:]]
			ds.addSample(microarray, (label))

def train(filename, start=''):
	load_params()
	#import dataset
	ds = ClassificationDataSet(micro_dim, 1, nb_classes=num_classes)
	extract_data(filename, ds)
	
	tr, val = ds.splitWithProportion(0.9) #10% validation data
	#softmax output layer
	tr._convertToOneOfMany()
	val._convertToOneOfMany()
	
	#build network
	if start == '':
		ann = buildNetwork(tr.indim, 48, 3, 48, tr.outdim, hiddenclass=SigmoidLayer, recurrent=False, outclass=SoftmaxLayer, bias=inc_bias)
		iteration = 0
	else:
		ann = NetworkReader.readFrom(start)
		match = re.search('([0-9]+)_(?:[0-9]{1,3}).xml', start)
		if match == None:
			print 'Net save files should be named I_E.xml, where I is the iteration and E is the rounded error from 0-100'
			exit(1)
		else:
			iteration = int(match.group(1)) + 1
	
	#training 
	trainer = BackpropTrainer(ann,	dataset=tr, momentum=momentum, weightdecay=weight_decay, verbose=True)
	done = False
	errors, variations = [], []
	
	while(not done):
		trainer.trainEpochs(num_epochs)
		
		"""nodes = [[], [], []]
		for c in ann.connections:
			index = -1
			if c._name == 'in':
				index = 0
			elif c._name == 'hidden0':
				index = 1
			elif c._name == 'out':
				index = 2
			if index >= 0:
				nodes[index] = c.inputbuffer.tolist()"""
		
		"""matrices = [[], []]
		for mod in ann.modules:
			for conn in ann.connections[mod]:
				if mod.name == 'in' and conn.outmod.name == 'hidden0':
					matrices[0] = array(conn.params).reshape(shapes[0]).tolist()
				elif mod.name == 'hidden0' and conn.outmod.name == 'out':
					matrices[1] = array(conn.params).reshape(shapes[1]).tolist()"""
		
		"""f, axs = plt.subplots(1, len(matrices), sharey=True)
		for i in range(len(matrices)):
			axs[i].imshow(matrices[i], interpolation='none', shape=(24, 12))
		f.savefig('matrices-%d.png' % iteration)"""
		#graph_visualizer(nodes, matrices, iteration)
		
		errors.append(calcError(trainer))
		variations.append(calcVariation(trainer))
		print 'iter %d, error %f, variation %f' % (iteration, errors[-1], variations[-1])

		if iteration >= max_iterations - 1:
			done = True
		
		#save every <snapshot> iterations
		if iteration % snapshot == 0:
			file_data = (iteration, int(errors[-1]*100))
			print 'Saving model %d_%d.xml...' % file_data
			NetworkWriter.writeToFile(ann, '%d_%d.xml' % file_data)
		
		iteration = iteration + 1
	
	#testing
	val_error, val_variation = calcError(trainer, dataset=val), calcVariation(trainer, dataset=val)
	print 'error %f, variation %f' % (val_error, val_variation)
	
	#plotting
	iterations = range(max_iterations)
	fig, ax1 = plt.subplots()
	ax1.plot(iterations, map(log10, errors), 'b-')
	ax1.set_xlabel('iteration')
	ax1.set_ylabel('mean squared error')
	for tick in ax1.get_yticklabels():
		tick.set_color('b')
	ax1.set_title('error for validation dataset: %f, variation for validation dataset: %f' % (val_error, val_variation))
	ax2 = ax1.twinx()
	ax2.plot(iterations, map(log10, variations), 'r-')
	ax2.set_ylabel('variation (L1 error)')
	for tick in ax2.get_yticklabels():
		tick.set_color('r')
	plt.savefig('error-5layer-48-3-48.png')

def graph_visualizer(vertices, edges, iteration):
	g = Graph(directed=False)
	vpos = g.new_vertex_property("vector<float>")
	vsize = g.new_vertex_property("float")
	vcolor = g.new_vertex_property("vector<float>")
	ecolor = g.new_edge_property("vector<float>")
	ewidth = g.new_edge_property("float")

	layers_vertices = []
	for i in range(len(vertices)):
		layer = vertices[i][0]
		layer_vertices = []
		offset = len(layer) / 2.0
		for j in range(len(layer)):
			vertex = g.add_vertex()
			layer_vertices.append(vertex)
			vpos[vertex] = [i * 10, j - offset]
			vsize[vertex] = layer[j] * 3
			vcolor[vertex] = [0, 105/255.0, 62/255.0, 1]
		layers_vertices.append(layer_vertices)

	for i in range(len(edges)):
		matrix = edges[i]
		for j in range(len(matrix)):
			vector = matrix[j]
			for k in range(len(vector)):
				edge = g.add_edge(layers_vertices[i][j], layers_vertices[i+1][k])
				shade = 1 - vector[k]
				ecolor[edge] = [shade, shade, shade, 1]
				ewidth[edge] = 2

	graph_draw(g, vpos, vertex_size=vsize, vertex_fill_color=vcolor, edge_color=ecolor, edge_pen_width=ewidth, bg_color=[1,1,1,1], output_size=(800, 800), output='graph-%d.png' % iteration)

def calcError(trainer, dataset=None):
    if dataset == None:
        dataset = trainer.ds
    dataset.reset()
    out = []
    targ = []
    for seq in dataset._provideSequences():
        trainer.module.reset()
        for input, target in seq:
            res = trainer.module.activate(input)
            out.append(argmax(res))
            targ.append(argmax(target))
    return percentError(out, targ) / 100

def calcVariation(trainer, dataset=None):
    if dataset == None:
        dataset = trainer.ds
    dataset.reset()
    errors = []
    for seq in dataset._provideSequences():
        trainer.module.reset()
        for input, target in seq:
            res = trainer.module.activate(input)
            error_sum = 0.
            for i in range(len(res)):
            	error_sum += abs(res[i] - target[i])
            errors.append(error_sum)
    return sum(errors) / len(errors)

if __name__ == '__main__':
	operation = sys.argv[1]
	if operation in 'train':
		train(sys.argv[2])
	else:
		train(sys.argv[2], sys.argv[3])

