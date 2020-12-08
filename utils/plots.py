import os
import pandas as pd
import numpy as np

os.environ["R_HOME"] = "R-3.6.2"
os.environ["R_USER"] = "/venv/Lib/site-packages/rpy2"

import rpy2.robjects.packages as rpackages
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

base = rpackages.importr("base")
utils = rpackages.importr("utils")

df = pd.read_csv("scenario1.csv")
data = df.to_numpy()
index = np.argmax(data[:, 0] == 5, axis=0)
index = index if index > 0 else len(data)
print(index)


def main():
    rpy2.robjects.numpy2ri.activate()
    utils.chooseCRANmirror(ind=1)
    person1 = ro.IntVector(data[:index, 1])
    person2 = ro.IntVector(data[:index, 2])
    ro.r(f'''
    #install.packages("igraph")
    #install.packages("visNetwork")
    #install.packages("networkD3")
    Sys.setenv(RSTUDIO_PANDOC="pandoc")
    s <- Sys.getenv("RSTUDIO_PANDOC")
    ''')
    print(ro.r.__getattribute__('s'))
    hierarchicalRepulsion(person1, person2)
    # plot_graph()


def get_coordinates():
    nodes = list(np.unique(np.append(np.unique(data[:index, 1]), np.unique(data[:index, 2]))))
    longitude = [None] * len(nodes)
    latitude = [None] * len(nodes)

    for x in data[:index]:
        if longitude[nodes.index(x[1])] is None:
            longitude[nodes.index(x[1])] = x[8]
        if latitude[nodes.index(x[1])] is None:
            latitude[nodes.index(x[1])] = x[7]
        if longitude[nodes.index(x[2])] is None:
            longitude[nodes.index(x[2])] = x[-1]
        if latitude[nodes.index(x[2])] is None:
            latitude[nodes.index(x[2])] = x[-2]

    return longitude, latitude


def hierarchicalRepulsion(person1, person2, nodeDistance=120, central_gravity=0.0, spring_length=100,
                          spring_constant=0.01, damping=0.09):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
              from=person1,
              to=person2
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(id = node_names, label = node_names)
            p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(physics = T) %>% visPhysics(solver = "hierarchicalRepulsion",
            hierarchicalRepulsion = list(nodeDistance = {nodeDistance}, centralGravity = {central_gravity},
            springLength = {spring_length}, springConstant = {spring_constant}, damping = {damping}))
            visSave(p, file = "test.html")
            ''')


def repulsion(person1, person2, nodeDistance=100, central_gravity=0.2, spring_length=200, spring_constant=0.05,
              damping=0.09):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
              from=person1,
              to=person2
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(id = node_names, label = node_names)
            p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(physics = T) %>% visPhysics(solver = "repulsion",
            repulsion = list(nodeDistance = {nodeDistance}, centralGravity = {central_gravity},
            springLength = {spring_length}, springConstant = {spring_constant}, damping = {damping}))
            visSave(p, file = "test.html")
            ''')


def barnesHut(person1, person2, gravity_constant=-2000, central_gravity=0.3, spring_length=95, spring_constant=0.04,
              damping=0.09, avoidOverlap=0):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
              from=person1,
              to=person2
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(id = node_names, label = node_names)
            p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(physics = T) %>% visPhysics(solver = "barnesHut",
            barnesHut = list(gravitationalConstant = {gravity_constant}, centralGravity = {central_gravity},
            springLength = {spring_length}, springConstant = {spring_constant}, damping = {damping},
            avoidOverlap = {avoidOverlap}))
            visSave(p, file = "test.html")
            ''')


def forced_atlas(person1, person2, gravity_constant=-50, central_gravity=0.01, spring_length=100, spring_constant=0.08,
                 damping=0.4, avoidOverlap=0):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r(f'''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
              from=person1,
              to=person2
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(id = node_names, label = node_names)
            p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(physics = T) %>% visPhysics(solver = "forceAtlas2Based",
            forceAtlas2Based = list(gravitationalConstant = {gravity_constant}, centralGravity = {central_gravity},
            springLength = {spring_length}, springConstant = {spring_constant}, damping = {damping},
            avoidOverlap = {avoidOverlap}))
            visSave(p, file = "test.html")
            ''')


def dh_layout(person1, person2):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r('''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
              from=person1,
              to=person2
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(id = node_names, label = node_names)
            p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(layout = "layout_with_dh")
            visSave(p, file = "test.html")
            ''')


def sphere_layout(person1, person2):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r('''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
              from=person1,
              to=person2
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(id = node_names, label = node_names)
            p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(layout = "layout_on_sphere")
            visSave(p, file = "test.html")
            ''')


def sugiyama_layout(person1, person2):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r('''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
              from=person1,
              to=person2
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(id = node_names, label = node_names)
            p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(layout = "layout_with_sugiyama")
            visSave(p, file = "test.html")
            ''')


def mds_layout(person1, person2):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r('''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
              from=person1,
              to=person2
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(id = node_names, label = node_names)
            p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(layout = "layout_with_mds")
            visSave(p, file = "test.html")
            ''')


def lgl_layout(person1, person2):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r('''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
              from=person1,
              to=person2
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(id = node_names, label = node_names)
            p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(layout = "layout_with_lgl")
            visSave(p, file = "test.html")
            ''')


def kk_layout(person1, person2):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r('''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
              from=person1,
              to=person2
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(id = node_names, label = node_names)
            p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(layout = "layout_with_kk")
            visSave(p, file = "test.html")
            ''')


def graphopt_layout(person1, person2):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r('''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
              from=person1,
              to=person2
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(id = node_names, label = node_names)
            p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(layout = "layout_with_graphopt")
            visSave(p, file = "test.html")
            ''')


def fr_layout(person1, person2):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r('''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
              from=person1,
              to=person2
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(id = node_names, label = node_names)
            p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(layout = "layout_with_fr")
            visSave(p, file = "test.html")
            ''')


def geo_layout(person1, person2):
    longitude, latitude = get_coordinates()
    print(longitude, latitude)
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r.assign("longitude", longitude)
    ro.r.assign("latitude", latitude)
    ro.r('''
    library(igraph)
    library(visNetwork)
    library(htmlwidgets)
    
    data <- data.frame(
      from=person1,
      to=person2
    )
    
    node_names <- factor(sort(unique(c(as.character(data$from), 
                                   as.character(data$to)))))
    
    map <- data.frame(
        names = node_names,
        x = longitude,
        y = latitude
    )
    
    g <- graph.data.frame(data, directed = FALSE, vertices = map)
    g.vis <- toVisNetworkData(g)
    
    lo <- as.matrix(map[,2:3])

    p <- visNetwork(g.vis$nodes, g.vis$edges, width="100%", height = 1080) %>% 
                    visOptions(highlightNearest = list(enabled = T, hover = T))
    
    visSave(p, file = "test.html")
    ''')


def star_layout(person1, person2):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r('''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
              from=person1,
              to=person2
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(id = node_names, label = node_names)
            p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(layout = "layout_as_star")
            visSave(p, file = "test.html")
            ''')


def grid_layout(person1, person2):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r('''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
              from=person1,
              to=person2
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(id = node_names, label = node_names)
            p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(layout = "layout_on_grid")
            visSave(p, file = "test.html")
            ''')


def tree_layout(person1, person2):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r('''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
              from=person1,
              to=person2
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(id = node_names, label = node_names)
            p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(layout = "layout_as_tree")
            visSave(p, file = "test.html")
            ''')


def gem_layout(person1, person2):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r('''
            library(igraph)
            library(visNetwork)

            data <- data.frame(
              from=person1,
              to=person2
            )
            node_names <- factor(sort(unique(c(as.character(data$from), 
                                           as.character(data$to)))))
            nodes <- data.frame(id = node_names, label = node_names)
            p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
            hover = T)) %>% visIgraphLayout(layout = "layout_with_gem")
            visSave(p, file = "test.html")
            ''')


def circle_layout(person1, person2):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r('''
        library(igraph)
        library(visNetwork)

        data <- data.frame(
          from=person1,
          to=person2
        )
        node_names <- factor(sort(unique(c(as.character(data$from), 
                                       as.character(data$to)))))
        nodes <- data.frame(id = node_names, label = node_names)
        p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visOptions(highlightNearest = list(enabled = T, 
        hover = T)) %>% visIgraphLayout(layout = "layout_in_circle")
        visSave(p, file = "test.html")
        ''')


def visGraph(person1, person2):
    ro.r.assign("person1", person1)
    ro.r.assign("person2", person2)
    ro.r('''
    library(igraph)
    library(visNetwork)
    
    data <- data.frame(
      from=person1,
      to=person2
    )
    node_names <- factor(sort(unique(c(as.character(data$from), 
                                   as.character(data$to)))))
    nodes <- data.frame(id = node_names, label = node_names)
    p <- visNetwork(nodes, data, width="100%", height = 1080) %>% visIgraphLayout() #%>% visConfigure(enabled = FALSE, filter = "")
    
    saveWidgetFix <- function (widget,file) {
      wd<-getwd()
      on.exit(setwd(wd))
      outDir<-dirname(file)
      file<-basename(file)
      setwd(outDir);
      saveWidget(widget,file=file, selfcontained = TRUE)
    }
    library(htmlwidgets)
    saveWidgetFix(p, file="test.html")
    ''')


def plot_graph():
    ro.r('''
    library(igraph)
    library(networkD3)
    
    data <- data.frame(
      from=c("A", "A", "B", "D", "C", "D", "E", "B", "C", "D", "K", "A", "M"),
      to=c("B", "E", "F", "A", "C", "A", "B", "Z", "A", "C", "A", "B", "K")
    )
    node_names <- factor(sort(unique(c(as.character(data$from), 
                                   as.character(data$to)))))
                                   
    nodes <- data.frame(name = node_names, group = 1)
    
    links <- data.frame(source = match(data$from, node_names) - 1, 
                    target = match(data$to, node_names) - 1)
                    
    p <- forceNetwork(links, nodes, Source = "source", Target = "target", NodeID = "name", Group = "group",
                    linkDistance = 10, charge = -900, fontSize = 14, fontFamily = "serif", opacity = 0.9, zoom = T,
                    opacityNoHover = 1)
    
    saveWidgetFix <- function (widget,file) {
      wd<-getwd()
      on.exit(setwd(wd))
      outDir<-dirname(file)
      file<-basename(file)
      setwd(outDir);
      saveWidget(widget,file=file, selfcontained = TRUE)
    }
    library(htmlwidgets)
    saveWidgetFix(p, file="test.html")
''')
    # p = r.__getattribute__('p')
    # return p


if __name__ == '__main__':
    main()
