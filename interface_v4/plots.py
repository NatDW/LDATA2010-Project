import os
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pylab import get_cmap
import scipy
import seaborn as sb

os.environ['R_HOME'] = 'C:\\Program Files\\R\\R-3.6.2'
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\R\\R-3.6.2\\bin\\x64\\'
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\R\\R-3.6.2\\'

import rpy2.robjects.packages as rpackages
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

base = rpackages.importr("base")
utils = rpackages.importr("utils")

df = pd.read_csv("ldata2010_small_dataset.csv")
data = df.to_numpy()
index = np.argmax(data[:, 0] == 100, axis=0)
index = index if index > 0 else len(data)

data = data[:index]

edgeList = data[:index, 1:3].tolist()
edgeList = [sorted(x) for x in edgeList]

edf = pd.DataFrame(edgeList)
weight_edf = edf.pivot_table(index=[0, 1], aggfunc='size')

weight_edges = []
for x in data:
    tmp = sorted([x[1], x[2]])
    weight_edges.append(weight_edf[tmp[0], tmp[1]])
np.asarray(weight_edges)

G = nx.Graph()
for i in range(len(weight_edf.values)):
    G.add_edge(weight_edf.index[i][0], weight_edf.index[i][1], weight=weight_edf.iloc[i])
matrix = nx.adjacency_matrix(G).A
nodes = nx.nodes(G)

cmap = get_cmap('magma', 5)
print(rgb2hex(cmap(4)))


def main():
    rpy2.robjects.numpy2ri.activate()
    utils.chooseCRANmirror(ind=1)
    person1 = ro.IntVector(data[:index, 1])
    person2 = ro.IntVector(data[:index, 2])
    ro.r(f'''
    #install.packages("igraph")
    #install.packages("visNetwork")
    #install.packages("networkD3")
    #install.packages("ggplot2")
    #install.packages("plotly")
    #install.packages("hrbrthemes")
    #install.packages("viridis")
    #install.packages("heatmaply")
    Sys.setenv(RSTUDIO_PANDOC="pandoc")
    s <- Sys.getenv("RSTUDIO_PANDOC")
    ''')
    # print(ro.r.__getattribute__('s'))
    # adjacency_matrix(nodes, matrix)
    # repulsion(person1, person2, spring_length=100, spring_constant=0, nodeDistance=100, damping=0.5)
    print(clusters_coordinates_encounter(11))
    # plot_graph()


def clustering_coefficient(G, weight=None):
    return nx.clustering(G, weight=weight)


def degree(G, weight=None):
    return nx.degree(G, weight=weight)


def degree_assortativity(G, weight=None):
    return nx.clustering(G, weight=weight)


def clusters_coordinates_home():
    nodes_longitude, nodes_latitude = get_coordinates()
    nodes_longitude = np.array(nodes_longitude).reshape(-1, 1)
    nodes_latitude = np.array(nodes_latitude).reshape(-1, 1)
    kmeans = KMeans(n_clusters=min(len(np.unique(nodes_longitude)), len(np.unique(nodes_latitude))))
    kmeans.fit(nodes_longitude, nodes_latitude)
    return kmeans.labels_


def clusters_coordinates_encounter(timestep):
    d = df.to_numpy()
    nodes_id = list(np.unique(np.append(np.unique(d[d[:, 0] == timestep, 1]), np.unique(d[d[:, 0] == timestep, 2]))))
    nodes_longitude, nodes_latitude = get_coordinates_encounter(timestep)
    nodes_longitude = np.array(nodes_longitude).reshape(-1, 1)
    nodes_latitude = np.array(nodes_latitude).reshape(-1, 1)
    kmeans = KMeans(n_clusters=min(len(np.unique(nodes_longitude)), len(np.unique(nodes_latitude))))
    kmeans.fit(nodes_longitude, nodes_latitude)
    return d[d[:, 0] == timestep, 1], d[d[:, 0] == timestep, 2], kmeans.labels_


def shortest_path(source, target, weight=None):

    nodes_id = np.array(list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2])))))
    edge_colors = np.array(['#cdcdcd'] * len(edgeList))
    nodes_colors = np.array(['#666666'] * len(nodes_id))

    try:
        path = nx.shortest_path(G, source, target, weight=weight)
        for i in range(0, len(path) - 1):
            e = sorted([path[i], path[i + 1]])
            if e in edgeList:
                mask = np.all(np.isin(edgeList, e), axis=1)
                edge_colors[np.where(mask)] = '#ffa500'
            nodes_colors[np.where(nodes_id == path[i + 1])] = '#ffd700'
        nodes_colors[np.where(nodes_id == path[0])] = '#ffd700'

        return nodes_colors, edge_colors
    except:
        return nodes_colors, edge_colors


def get_infected():
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    infected = [None] * len(nodes_id)

    for x in reversed(data[:index]):
        if infected[nodes_id.index(x[1])] is None:
            infected[nodes_id.index(x[1])] = x[3]
        if infected[nodes_id.index(x[2])] is None:
            infected[nodes_id.index(x[2])] = x[4]

    return infected


def get_coordinates():
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    longitude = [None] * len(nodes_id)
    latitude = [None] * len(nodes_id)

    for x in data[:index]:
        if longitude[nodes_id.index(x[1])] is None:
            longitude[nodes_id.index(x[1])] = x[8]
        if latitude[nodes_id.index(x[1])] is None:
            latitude[nodes_id.index(x[1])] = x[7]
        if longitude[nodes_id.index(x[2])] is None:
            longitude[nodes_id.index(x[2])] = x[-1]
        if latitude[nodes_id.index(x[2])] is None:
            latitude[nodes_id.index(x[2])] = x[-2]

    return longitude, latitude


def get_coordinates_encounter(timestep):
    d = df.to_numpy()
    nodes_id = list(np.unique(np.append(np.unique(d[d[:, 0] == timestep, 1]), np.unique(d[d[:, 0] == timestep, 2]))))
    longitude = [None] * len(nodes_id)
    latitude = [None] * len(nodes_id)

    for x in reversed(d[d[:, 0] == timestep]):
        if longitude[nodes_id.index(x[1])] is None:
            longitude[nodes_id.index(x[1])] = x[6]
        if latitude[nodes_id.index(x[1])] is None:
            latitude[nodes_id.index(x[1])] = x[5]
        if longitude[nodes_id.index(x[2])] is None:
            longitude[nodes_id.index(x[2])] = x[6]
        if latitude[nodes_id.index(x[2])] is None:
            latitude[nodes_id.index(x[2])] = x[5]

    return longitude, latitude


def adjacency_matrix(nodes_list, adj_matrix):
    ro.r.assign("nodes_list", nodes_list)
    ro.r.assign("adj_matrix", adj_matrix)
    try:
        ro.r('''
                library(ggplot2)
                library(plotly)
                library(heatmaply)
                library(hrbrthemes)
                library(viridis)
                library(htmlwidgets)
                
                rownames(adj_matrix) <- nodes_list
                colnames(adj_matrix) <- nodes_list
                
                map <- heatmaply(adj_matrix,
                   hide_colorbar = FALSE,
                   dendrogram = FALSE,
                   label_names = c("Person 1", "Person 2", "Value"),
                   showticklabels = FALSE,
                   file = "test.html"
                )
        ''')
    except:
        """f = open("test.html", 'w')

        message = <html>
            <head></head>
            <body><p>There is not enough memory for containing the dataset</p></body>
            </html>

        f.write(message)
        f.close()"""


def hierarchicalRepulsion(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
                          path=None, weight=None, nodeDistance=120, central_gravity=0.0, spring_length=100,
                          spring_constant=0.01, damping=0.09):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)
    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)

            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%",
                height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
                hover = T)) %>% visIgraphLayout(physics = T) %>% visPhysics(solver = "hierarchicalRepulsion",
                hierarchicalRepulsion = list(nodeDistance = {nodeDistance}, centralGravity = {central_gravity},
                springLength = {spring_length}, springConstant = {spring_constant}, damping = {damping}))
            visSave(p, file = "test.html")
            ''')


def repulsion(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
              path=None, weight=None, nodeDistance=100, central_gravity=0.2, spring_length=200,
              spring_constant=0.05, damping=0.09):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)

            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%",
                height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
                hover = T)) %>% visIgraphLayout(physics = T) %>% visPhysics(solver = "repulsion",
                repulsion = list(nodeDistance = {nodeDistance}, centralGravity = {central_gravity},
                springLength = {spring_length}, springConstant = {spring_constant}, damping = {damping}))
            visSave(p, file = "test.html")
            ''')


def barnesHut(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
              path=None, weight=None, gravity_constant=-2000, central_gravity=0.3, spring_length=95,
              spring_constant=0.04, damping=0.09, avoidOverlap=0):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)

            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%",
            height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(physics = T) %>% visPhysics(solver = "barnesHut",
            barnesHut = list(gravitationalConstant = {gravity_constant}, centralGravity = {central_gravity},
            springLength = {spring_length}, springConstant = {spring_constant}, damping = {damping},
            avoidOverlap = {avoidOverlap}))
            visSave(p, file = "test.html")
            ''')


def forced_atlas(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
                 path=None, weight=None, gravity_constant=-50, central_gravity=0.01, spring_length=100,
                 spring_constant=0.08, damping=0.4, avoidOverlap=0):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)

            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%",
            height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(physics = T) %>% visPhysics(solver = "forceAtlas2Based",
            forceAtlas2Based = list(gravitationalConstant = {gravity_constant}, centralGravity = {central_gravity},
            springLength = {spring_length}, springConstant = {spring_constant}, damping = {damping},
            avoidOverlap = {avoidOverlap}))
            visSave(p, file = "test.html")
            ''')


def dh_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
              path=None, weight=None):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)
            
            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%", height = 1080) %>% 
            visOptions(highlightNearest = list(enabled = T, hover = T)) %>% visIgraphLayout(layout = "layout_with_dh")
            visSave(p, file = "test.html")
            ''')


def sphere_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
                  path=None, weight=None):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)
            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%", height = 1080) %>% 
            visOptions(highlightNearest = list(enabled = T, hover = T)) %>% visIgraphLayout(layout = "layout_on_sphere")
            visSave(p, file = "test.html")
            ''')


def sugiyama_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
                    path=None, weight=None):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)
            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%", height = 1080) %>%
            visOptions(highlightNearest = list(enabled = T, hover = T)) %>%
            visIgraphLayout(layout = "layout_with_sugiyama")
            visSave(p, file = "test.html")
            ''')


def mds_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
               path=None, weight=None):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)
            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%", height = 1080) %>%
            visOptions(highlightNearest = list(enabled = T, hover = T)) %>% visIgraphLayout(layout = "layout_with_mds")
            visSave(p, file = "test.html")
            ''')


def lgl_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
               path=None, weight=None):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)
            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%", height = 1080) %>%
            visOptions(highlightNearest = list(enabled = T, hover = T)) %>% 
            visIgraphLayout(layout = "layout_with_lgl", type = "full")
            visSave(p, file = "test.html")
            ''')


def kk_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
              path=None, weight=None):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)
            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%", height = 1080) %>%
            visOptions(highlightNearest = list(enabled = T, hover = T)) %>% visIgraphLayout(layout = "layout_with_kk")
            visSave(p, file = "test.html")
            ''')


def graphopt_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
                    path=None, weight=None):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)
            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%", height = 1080) %>%
            visOptions(highlightNearest = list(enabled = T, hover = T)) %>%
            visIgraphLayout(layout = "layout_with_graphopt")
            visSave(p, file = "test.html")
            ''')


def fr_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
              path=None, weight=None):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)
            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%", height = 1080) %>%
            visOptions(highlightNearest = list(enabled = T, hover = T)) %>% visIgraphLayout(layout = "layout_with_fr")
            visSave(p, file = "test.html")
            ''')


def geo_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
               path=None, weight=None):

    longitude, latitude = get_coordinates()
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r.assign("longitude", longitude)
    ro.r.assign("latitude", latitude)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
                x = longitude,
                y = latitude
            )
    
            g <- graph.data.frame(data, directed = FALSE, vertices = map)
            g.vis <- toVisNetworkData(g)
    
            lo <- as.matrix(map[, 11:12])

            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%", height = 1080) %>% 
            visOptions(highlightNearest = list(enabled = T, hover = T))
    
            visSave(p, file = "test.html")
    ''')


def star_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
                path=None, weight=None):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)
            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%", height = 1080) %>%
            visOptions(highlightNearest = list(enabled = T, hover = T)) %>% visIgraphLayout(layout = "layout_as_star")
            visSave(p, file = "test.html")
            ''')


def grid_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
                path=None, weight=None):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)
            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%", height = 1080) %>%
            visOptions(highlightNearest = list(enabled = T, hover = T)) %>% visIgraphLayout(layout = "layout_on_grid")
            visSave(p, file = "test.html")
            ''')


def tree_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
                path=None, weight=None):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)
            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%", height = 1080) %>%
            visOptions(highlightNearest = list(enabled = T, hover = T)) %>% visIgraphLayout(layout = "layout_as_tree")
            visSave(p, file = "test.html")
            ''')


def gem_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
               path=None, weight=None):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)
            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%", height = 1080) %>%
            visOptions(highlightNearest = list(enabled = T, hover = T)) %>% visIgraphLayout(layout = "layout_with_gem")
            visSave(p, file = "test.html")
        ''')


def circle_layout(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
                  path=None, weight=None):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)
            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%", height = 1080) %>%
            visOptions(highlightNearest = list(enabled = T, hover = T)) %>% visIgraphLayout(layout = "layout_in_circle")
            visSave(p, file = "test.html")
        ''')


def visGraph(person1, person2, color_map=None, edge_width=None, nodes_size=None, infected=None, group=None,
             path=None, weight=None):

    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)

    color_edge = np.array(["#7599c8"] * len(person1))

    if path is not None:
        color_map, color_edge = shortest_path(path[0], path[1], weight=weight)

    mask = '#'
    nodes_id = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    color_border = np.array([None] * len(nodes_id))
    if color_map is None:
        color_map = np.array(["#97c2fc"] * len(nodes_id))
    if edge_width is None:
        edge_width = np.array([1] * len(person1))
    if nodes_size is None:
        nodes_size = np.array([25] * len(nodes_id))
    if group is None:
        group = np.array([1] * len(nodes_id))
        mask = ''

    if infected is None:
        color_border = np.array(["#7599c8"] * len(nodes_id))
    else:
        for i in range(len(nodes_id)):
            if infected[i] == 0:
                color_border[i] = "lightgreen"
            else:
                color_border[i] = "darkred"

    border_width = np.ceil(np.divide(nodes_size, 5))

    color_edge = np.array(color_edge)
    color_map = np.array(color_map)
    edge_width = np.array(edge_width)
    nodes_size = np.array(nodes_size)
    group = np.array(group)

    ro.r.assign("color_map", color_map)
    ro.r.assign("edge_width", edge_width)
    ro.r.assign("color_edge", color_edge)
    ro.r.assign("nodes_size", nodes_size)
    ro.r.assign("border_width", border_width)
    ro.r.assign("color_border", color_border)
    ro.r.assign("group", group)

    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
                from=person1,
                to=person2,
                width=edge_width,
                color = color_edge
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(
                names = node_names,
                size = nodes_size,
                {mask}color.background = color_map,
                {mask}color.border = color_border,
                {mask}color.highlight.background = color_map,
                {mask}color.highlight.border = color_border,
                {mask}color.hover.background = color_map,
                {mask}color.hover.border = color_border,
                group = group,
                borderWidth = border_width
            )

            g <- graph.data.frame(data, directed = FALSE, vertices = nodes)
            g.vis <- toVisNetworkData(g)
            p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%", height = 1080) %>% 
            visOptions(highlightNearest = list(enabled = T, hover = T)) %>%
            visIgraphLayout()
            visSave(p, file = "test.html")
    ''')


if __name__ == '__main__':
    main()