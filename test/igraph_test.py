import re, sys
import igraph

# Construct a graph with 5 vertices
n_vertices = 5
edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (3, 4)]
g = igraph.Graph(n_vertices, edges)

# Set attributes for the graph, nodes, and edges
g["title"] = "Small Social Network"
g.vs["name"] = [
    "Daniel Morillas", "Kathy Archer", "Kyle Ding", "Joshua Walton",
    "Jana Hoyer"
]
g.vs["gender"] = ["M", "F", "F", "M", "F"]

# Set individual attributes
g.vs[1]["name"] = "Kathy Morillas"
