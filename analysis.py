# Henry Pu
# S01408737

import numpy
import matplotlib
import random
import matplotlib.pyplot as plt
import pylab
import types
import time
import math
import copy
def compute_largest_cc_size(g: list) -> int:
    visited=set()
    largest_cc=0
    for node in g.keys():
        if node not in visited:
            current_cc=graph_size(g,node, visited)
            if current_cc > largest_cc:
                largest_cc=current_cc
    return largest_cc
def graph_size(graph, start_node, visited):
    queue = [start_node]
    size = 0
    while len(queue) != 0:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            size += 1
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
    return size
def read_graph(filename):
    """
    Read a graph from a file.  The file is assumed to hold a graph
    that was written via the write_graph function.
    Arguments:
    filename -- name of file that contains the graph
    Returns:
    The graph that was stored in the input file.
    """
    with open(filename) as f:
        g = eval(f.read())
    return g
def write_graph(g, filename):
    """
    Write a graph to a file.  The file will be in a format that can be
    read by the read_graph function.
    Arguments:
    g        -- a graph
    filename -- name of the file to store the graph
    Returns:
    None
    """
    with open(filename, 'w') as f:
        f.write(repr(g))
def copy_graph(g):
    """
    Return a copy of the input graph, g
    Arguments:
    g -- a graph
    Returns:
    A copy of the input graph that does not share any objects.
    """
    return copy.deepcopy(g)
## Timing functions
def time_func(f, args=[], kw_args={}):
    """
    Times one call to f with args, kw_args.
    Arguments:
    f       -- the function to be timed
    args    -- list of arguments to pass to f
    kw_args -- dictionary of keyword arguments to pass to f.
    Returns:
    a tuple containing the result of the call and the time it
    took (in seconds).
Example:
    >>> def sumrange(low, high):
            sum = 0
            for i in range(low, high):
                sum += i
            return sum
    >>> time_func(sumrange, [82, 35993])
    (647726707, 0.01079106330871582)
    >>>
    """
    start_time = time.time()
    result = f(*args, **kw_args)
    end_time = time.time()
    return (result, end_time - start_time)
## Plotting functions

def show():
    """
    Do not use this function unless you have trouble with figures.
    It may be necessary to call this function after drawing/plotting
    all figures.  If so, it should only be called once at the end.
    Arguments:
    None
    Returns:
    None
    """
    plt.show()
def plot_dist_linear(data, title, xlabel, ylabel, filename=None):
    """
    Plot the distribution provided in data as a bar plot on a linear
    scale.
Arguments:
data
-- dictionary which will be plotted with the keys
   on the x axis and the values on the y axis
-- title label for the plot
-- x axis label for the plot
-- y axis label for the plot
title
xlabel
ylabel
filename -- optional name of file to which plot will be
                saved (in png format)
    Returns:
    None
    """
    _plot_dist(data, title, xlabel, ylabel, False, filename)
def plot_dist_loglog(data, title, xlabel, ylabel, filename=None):
    """
    Plot the distribution provided in data as a scatter plot on a
    loglog scale.
Arguments:
data
-- dictionary which will be plotted with the keys
   on the x axis and the values on the y axis
-- title label for the plot
-- x axis label for the plot
-- y axis label for the plot
title
xlabel
ylabel
filename -- optional name of file to which plot will be
                saved (in png format)
    Returns:
    None
    """
    _plot_dist(data, title, xlabel, ylabel, True, filename)
def _pow_10_round(n, up=True):
    """
    Round n to the nearest power of 10.

    Arguments:
    n  -- number to round
    up -- round up if True, down if False
    Returns:
    rounded number
    """
    if up:
        return 10 ** math.ceil(math.log(n, 10))
    else:
        return 10 ** math.floor(math.log(n, 10))
def _plot_dist(data, title, xlabel, ylabel, scatter, filename=None):
    """
    Plot the distribution provided in data.
    Arguments:
    data
    -- dictionary which will be plotted with the keys
    on the x axis and the values on the y axis
    -- title label for the plot
    -- x axis label for the plot
    -- y axis label for the plot
    title
    xlabel
    ylabel
    scatter  -- True for loglog scatter plot, False for linear bar plot
    filename -- optional name of file to which plot will be
                saved (in png format)
    Returns:
    None
    """
    ### Check that the data is a dictionary
    if not isinstance(data, dict):
        msg = "data must be a dictionary, not {0}".format(type(data).__name__)
        raise TypeError(msg)
    ### Create a new figure
    fig = pylab.figure()
    ### Plot the data
    if scatter:
        _plot_dict_scatter(data)
    else:
        _plot_dict_bar(data, 0)
    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    ### Draw grid
    gca = pylab.gca()
    gca.yaxis.grid(True)
    gca.xaxis.grid(False)
    if scatter:
        ### Use loglog scale
        gca.set_xscale('log')
        gca.set_yscale('log')
        gca.set_xlim([_pow_10_round(min([x for x in data.keys() if x > 0]), False),

                _pow_10_round(max(data.keys()))])
    gca.set_ylim([_pow_10_round(min([x for x in data.values() if x > 0]),
                
    False),_pow_10_round(max(data.values()))])
    ### Show the plot
    fig.show()
    ### Save to file
    if filename:
        pylab.savefig(filename)
def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
    """
    Plot a line graph with the provided data.
    Arguments:
    data
    title
    xlabel
    ylabel
    labels
    -- a list of dictionaries, each of which will be plotted
    as a line with the keys on the x axis and the values on
    the y axis.
    -- title label for the plot
    -- x axis label for the plot
    -- y axis label for the plot
    -- optional list of strings that will be used for a legend
                this list must correspond to the data list
    filename -- optional name of file to which plot will be
                saved (in png format)
    Returns:
    None
    """
    ### Check that the data is a list
    if not isinstance(data, list):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)
    ### Create a new figure
    fig = pylab.figure()
    ### Plot the data
    if labels:
        mylabels = labels[:]
        for _ in range(len(data)-len(labels)):
            mylabels.append("")
        for d, l in zip(data, mylabels):
            _plot_dict_line(d, l)
        # Add legend
        pylab.legend(loc='best')
        gca = pylab.gca()
        legend = gca.get_legend()
        pylab.setp(legend.get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)
    ### Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    ymin = min(0, min(mins))
    pylab.ylim(ymin=ymin)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    ### Draw grid lines
    pylab.grid(True)
    ### Show the plot
    fig.show()
    ### Save to file
    if filename:
        pylab.savefig(filename)
def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.
    Arguments:
    data -- dictionary
    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = list(data.keys())
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals
def _plot_dict_line(d, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.
    Arguments:
    d     -- dictionary
    label -- optional legend label
    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        pylab.plot(xvals, yvals, label=label)
    else:
        pylab.plot(xvals, yvals)
def _plot_dict_bar(d, xmin=None, label=None):
    """
    Plot data in the dictionary d on the current plot as bars.
    Arguments:
    d     -- dictionary
    xmin  -- optional minimum value for x axis
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if xmin == None:
        xmin = min(xvals) - 1
    else:
        xmin = min(xmin, min(xvals) - 1)
    if label:
        pylab.bar(xvals, yvals, align='center', label=label)
        pylab.xlim([xmin, max(xvals)+1])
    else:
        pylab.bar(xvals, yvals, align='center')
        pylab.xlim([xmin, max(xvals)+1])
def _plot_dict_scatter(d):
    """
    Plot data in the dictionary d on the current plot as points.
    Arguments:
    d     -- dictionary
    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    pylab.scatter(xvals, yvals)
def upa(n, m):
    """
    Generate an undirected graph with n node and m edges per node
    using the preferential attachment algorithm.
    Arguments:
    n -- number of nodes
    m -- number of edges per node
    Returns:
    undirected random graph in UPAG(n, m)
    """
    g = {}
    if m <= n:
        g = make_complete_graph(m)
        for new_node in range(m, n):
            # Find <=m nodes to attach to new_node
            totdeg = float(total_degree(g))
            nodes = list(g.keys())
            probs = []
            for node in nodes:
                probs.append(len(g[node]) / totdeg)
            mult = distinct_multinomial(m, probs)
            # Add new_node and its random neighbors
            g[new_node] = set()
            for idx in mult:
                node = nodes[idx]
                g[new_node].add(node)
                g[node].add(new_node)
    return g

def erdos_renyi(n, p):
    """
    Generate a random Erdos-Renyi graph with n nodes and edge probability p.
    Arguments:
    n -- number of nodes
    p -- probability of an edge between any pair of nodes
    Returns:
    undirected random graph in G(n, p)
    """
    g = {}
    ### Add n nodes to the graph
    for node in range(n):
        g[node] = set()
    ### Iterate through each possible edge and add it with
    ### probability p.
    for u in range(n):
        for v in range(u+1, n):
            r = random.random()
            if r < p:
                g[u].add(v)
                g[v].add(u)
    return g
def total_degree(g):
    """
    Compute total degree of the undirected graph g.
    Arguments:
    g -- undirected graph
    Returns:
    Total degree of all nodes in g
    """
    return sum(map(len, g.values()))
def make_complete_graph(num_nodes):
    """
    Returns a complete graph containing num_nodes nodes.
    The nodes of the returned graph will be 0...(num_nodes-1) if num_nodes-1 is
positive.
    An empty graph will be returned in all other cases.
    Arguments:
    num_nodes -- The number of nodes in the returned graph.
    Returns:
    A complete graph in dictionary form.
    """
    result = {}
    for node_key in range(num_nodes):

            result[node_key] = set()
            for node_value in range(num_nodes):
                if node_key != node_value:
                    result[node_key].add(node_value)
    return result
def distinct_multinomial(ntrials, probs):
    """
    Draw ntrials samples from a multinomial distribution given by
    probs.  Return a list of indices into probs for all distinct
    elements that were selected.  Always returns a list with between 1
    and ntrials elements.
    Arguments:
    ntrials -- number of trials
    probs   -- probability vector for the multinomial, must sum to 1
    Returns:
    A list of indices into probs for each element that was chosen one
    or more times.  If an element was chosen more than once, it will
    only appear once in the result.
    """
    ### select ntrials elements randomly
    mult = numpy.random.multinomial(ntrials, probs)
    ### turn the results into a list of indices without duplicates
    result = [i for i, v in enumerate(mult) if v > 0]
    return result
network=read_graph("rf7.repr")
#number of nodes and edges
num_edges = 0
for vertex in network:
    num_edges += len(network[vertex])
# divide by 2 since each edge is counted twice
num_edges //= 2
n=len(network)
m=int(total_degree(network)/2/(len(network)))
#initialize upa and erdos graphs
undirected=upa(n,m)
erdos=erdos_renyi(n,total_degree(network)/n/(n-1))
def random_attack(graph):
    cc_size={}
    num_nodes_to_remove = (len(graph) * .2)
    graph_copy = copy_graph(graph)
    count=0
    while len(graph_copy) > (len(graph) - num_nodes_to_remove):
        # get a random node from the graph and remove it
        node_to_remove = random.choice(list(graph_copy.keys()))
        graph_copy.pop(node_to_remove)

# remove any edges that connect to the removed node
        for node, edges in graph_copy.items():
            if node_to_remove in edges:
                edges.remove(node_to_remove)
        y=compute_largest_cc_size(graph_copy)
        count+=1
        cc_size[count]=y
    return cc_size
def targeted_attack(graph):
    num_nodes_to_remove = (len(graph) * .2)
    graph_copy=copy_graph(graph)
    cc_size={}
    count=0
    while len(graph_copy) > (len(graph) - num_nodes_to_remove):
        # get a list of nodes sorted by degree in decreasing order
        sorted_nodes = sorted(graph_copy, key=lambda node: len(graph[node]),reverse=True)
        # remove the node with the highest degree and any edges that connect to it
        node_to_remove = sorted_nodes[0]
        for node, edges in graph.items():
            if node == node_to_remove:
                graph_copy.pop(node)
            elif node_to_remove in edges:
                edges.remove(node_to_remove)
        # remove any edges that connect to the removed node
        for node, edges in graph_copy.items():
            if node_to_remove in edges:
                edges.remove(node_to_remove)
        y=compute_largest_cc_size(graph_copy)
        count+=1
        cc_size[count]=y
    return cc_size
random_network=random_attack(network)
targeted_network=targeted_attack(network)
random_erdos=random_attack(erdos)
targeted_erdos=targeted_attack(erdos)
random_undirected=random_attack(undirected)
targeted_undirected=targeted_attack(undirected)
plot_lines([random_network,targeted_network,random_erdos,targeted_erdos,random_undirected,targeted_undirected],
           "Size of Largest Connect Component vs Number of nodes removed","Nodes Removed","Size of Largest Connected Component",
           ["random network","targeted network","random erdos","targeted erdos","random upa","targetedupa"])
show()