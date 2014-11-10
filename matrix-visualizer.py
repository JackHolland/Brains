import matplotlib.pyplot as plt
import numpy as np
import random as rand

def random_matrix(i):
	matrix = np.zeros(shape=(100, 20))
	for i in range(np.shape(matrix)[0]):
		for j in range(np.shape(matrix)[1]):
			matrix[i, j] = (i + j) % 3
	return matrix

edges = map(random_matrix, range(3))

f, axs = plt.subplots(1, len(edges), sharey=True)
for i in range(len(edges)):
	axs[i].imshow(edges[i], interpolation='none')
f.savefig("matrices.png")
