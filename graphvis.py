from graph_tool.all import *

vertices = [[1, 2, 1], [2, 3], [4]]
edges = [
	[[0.5, 0.8], [0.2, 0.3], [0.8, 0.4]],
	[[0.3], [0.6]]
]

g = Graph(directed=False)
vpos = g.new_vertex_property("vector<float>")
vsize = g.new_vertex_property("float")
vcolor = g.new_vertex_property("vector<float>")
ecolor = g.new_edge_property("vector<float>")
ewidth = g.new_edge_property("float")

layers_vertices = []
for i in range(len(vertices)):
	layer = vertices[i]
	layer_vertices = []
	offset = len(layer) / 2.0
	for j in range(len(layer)):
		vertex = g.add_vertex()
		layer_vertices.append(vertex)
		vpos[vertex] = [i, j - offset]
		vsize[vertex] = layer[j] * 5
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

graph_draw(g, vpos, vertex_size=vsize, vertex_fill_color=vcolor, edge_color=ecolor, edge_pen_width=ewidth, bg_color=[1,1,1,1], output="graph.png")
