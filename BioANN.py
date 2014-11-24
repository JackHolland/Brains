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
from scipy import array, where, argmax, shape
import matplotlib.pyplot as plt
import re
from graph_tool.all import *
from math import log10

#Default parameters
micro_dim = 24 #the number of input nodes
num_classes = 3
num_hidden = [48,12] #number of hidden layers
shapes = [(24, 48), (48, 12)]
inc_bias = True #to include bias or not

momentum = 0.1 #momentum 
weight_decay = 0.1 #weight decay

myError = 10.0


max_iterations = 1000000 #maximum number of iterations
num_epochs = 5 #number of ephochs at every iteration
snapshot = 10 #when to snapshot the ANN

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
	shapes = [(micro_dim, num_hidden[0])]
	for i in range(1, len(num_hidden)-1):
		shapes.append((num_hidden[i], num_hidden[i+1]))
	shapes.append((num_hidden[-1], 3))
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

def train(data_file, vis_matrix, vis_graph, save_file=''):
	load_params()
	#import dataset
	ds = ClassificationDataSet(micro_dim, 1, nb_classes=num_classes)
	extract_data(data_file, ds)
	
	tr, val = ds.splitWithProportion(2/3.)
	#softmax output layer
	tr._convertToOneOfMany()
	val._convertToOneOfMany()
	
	#build network
	if save_file == '':
		ann = buildNetwork(tr.indim, num_hidden[0], tr.outdim, hiddenclass=SigmoidLayer, recurrent=False, outclass=SoftmaxLayer, bias=inc_bias)
		iteration = 0
	else:
		ann = NetworkReader.readFrom(save_file)
		match = re.search('([0-9]+)_(?:[0-9]{1,3}).xml', save_file)
		if match == None:
			print 'Net save files should be named I_E.xml, where I is the iteration and E is the rounded error from 0-100'
			exit(1)
		else:
			iteration = int(match.group(1)) + 1
	
	#training 
	trainer = BackpropTrainer(ann,	dataset=tr, momentum=momentum, weightdecay=weight_decay)
	done = False
	errors, variations = [], []
	testing_errors, testing_variations = [], []
	
	while(not done):
		trainer.trainEpochs(num_epochs)
		
		# visualize iteration
		if vis_matrix or vis_graph:
			vertices, edges = vertsEdges(ann)
			if vis_matrix:
				matrixVisualizer(edges)
			if vis_graph:
				graphVisualizer(vertices, edges, iteration)
		
		# calculate and print error info
		training_error, testing_error, training_variation, testing_variation = trainer.testOnData(), trainer.testOnData(dataset=val), calcVariation(trainer), calcVariation(trainer, dataset=val)
		errors.append(training_error)
		variations.append(training_variation)
		testing_errors.append(testing_error)
		testing_variations.append(testing_variation)
		fig, ax1 = plt.subplots()
		iterations = range(iteration+1)
		ax1.plot(iterations, map(log10, errors), 'r-')
		ax1.plot(iterations, map(log10, testing_errors), 'b-')
		ax1.set_xlabel('iteration')
		ax1.set_ylabel('log mean squared error (red=train, blue=test)')
		for tick in ax1.get_yticklabels():
			tick.set_color('b')
		ax2 = ax1.twinx()
		ax2.plot(iterations, map(log10, variations), 'r--')
		ax2.plot(iterations, map(log10, testing_variations), 'b--')
		ax2.set_ylabel('log variation (L1 error) (red=train, blue=test)')
		for tick in ax2.get_yticklabels():
			tick.set_color('r')
		plt.savefig('error-3layer-48.pdf')
		plt.close()
		print 'iter %d, training error %f, testing error %f, training variation %f, testing variation %f' % (iteration, training_error, testing_error, training_variation, testing_variation)
		
		#save every <snapshot> iterations
		if iteration % snapshot == -1:
			file_data = (iteration, int(errors[-1]*100))
			print 'Saving model %d_%d.xml...' % file_data
			NetworkWriter.writeToFile(ann, '%d_%d.xml' % file_data)
		
		# go to the next iteration if not done
		iteration = iteration + 1
		if iteration >= max_iterations:
			done = True
	
	#testing
	val_errors, val_variations = [], []
	for i in range(5):
		val_error, val_variation = trainer.testOnData(dataset=val), calcVariation(trainer, dataset=val)
		print 'error %f, variation %f' % (val_error, val_variation)
		val_errors.append(val_error)
		val_variations.append(val_variation)
		tr, val = ds.splitWithProportion(0.9)
		val._convertToOneOfMany()
	print 'average error %f, average variation %f' % (np.average(val_errors), np.average(val_variations))
	
	#plotting
	iterations = range(max_iterations)
	fig, ax1 = plt.subplots()
	ax1.plot(iterations, map(log10, errors), 'b-')
	ax1.set_xlabel('iteration')
	ax1.set_ylabel('log mean squared error')
	for tick in ax1.get_yticklabels():
		tick.set_color('b')
	ax1.set_title('error for validation dataset: %f, variation for validation dataset: %f' % (val_error, val_variation))
	ax2 = ax1.twinx()
	ax2.plot(iterations, map(log10, variations), 'r-')
	ax2.set_ylabel('log variation (L1 error)')
	for tick in ax2.get_yticklabels():
		tick.set_color('r')
	plt.savefig('error-4layer-48-96.pdf')

def vertsEdges(ann):
	num_layers = len(num_hidden) + 2
	nodes = []
	for i in range(num_layers):
		nodes.append([])
	for c in ann.connections:
		index = -1
		if c._name == 'in':
			index = 0
		elif c._name[:-1] == 'hidden':
			index = 1 + int(c._name[-1])
		elif c._name == 'out':
			index = num_layers - 1
		if index >= 0:
			nodes[index] = c.inputbuffer.tolist()
	edges = []
	for i in range(num_layers - 1):
		edges.append([])
	for mod in ann.modules:
		for conn in ann.connections[mod]:
			layer1, layer2 = mod.name, conn.outmod.name
			if layer1 == 'in' and layer2[:-1] == 'hidden':
				index = 0
			elif layer1[:-1] == 'hidden' and layer2 == 'out':
				index = num_layers - 2
			elif layer1 != 'bias':
				index = int(layer1[-1])
			if layer1 != 'bias':
				print index, shapes, shape(array(conn.params))
				edges[index] = array(conn.params).reshape(shapes[index]).tolist()
	return nodes, edges

def matrixVisualizer(edges):
	f, axs = plt.subplots(1, len(edges), sharey=True)
	for i in range(len(edges)):
		axs[i].imshow(edges[i], interpolation='none', shape=shapes[i])
	f.savefig('matrices-%d.png' % iteration)

def graphVisualizer(vertices, edges, iteration):
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

def param_true(param):
	return param == '1' or param == 'true'

if __name__ == '__main__':
	num_args = len(sys.argv)
	if num_args == 1:
		print 'usage: BioANN.py data_file [visualize_matrix visualize_graph save_file]'
	data_file, vis_matrix, vis_graph, save_file = sys.argv[1], False, False, ''
	if num_args >= 3:
		vis_matrix = param_true(sys.argv[2])
	if num_args >= 4:
		vis_graph = param_true(sys.argv[3])
	if num_args >= 5:
		save_file = sys.argv[4]
	train(data_file, vis_matrix, vis_graph, save_file)

