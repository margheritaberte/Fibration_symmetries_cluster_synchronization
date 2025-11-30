import networkx as nx
import networkx.algorithms.bipartite as bipartite
import numpy as np

from hypergraphx.core.hypergraph import Hypergraph
from hypergraphx.representations.projections import bipartite_projection


def kameicock_coloring(G, initial_types=None):
    """
    Color a graph using Kamei & Cock's algorithm, starting from the configuration given in initial_types.
    
    Parameters:
        G: A NetworkX graph
        initial_types: A dictionary of {node: initial_color}
    Return:
        A dictionary of {node: color} for all nodes
    """

    # In case of bipartite graph, the nodes of the two layers will have different initial types

    if initial_types is None:
        initial_types = {}
    
    # Initialize colors with initial types
    c = initial_types.copy()
    
    # Find the next available color
    next_color = max(initial_types.values(), default=0) + 1
    
    # Assign next_color to nodes not in initial_types
    for node in G.nodes():
        if node not in c:
            c[node] = next_color
    
    N = len(G)
    #j = 0
    Nj = max(c.values())
    
    while True:
        # Calculate for each node the number of neighbors of all the colors
        # Ii is a tuple of length Nj where in position k has the number of neighbors with color k
        I = []
        for node in G.nodes():
            Ii = tuple(sum(1 for neighbor in G[node] if c[neighbor] == k) for k in range(Nj+1))
            I.append(Ii)
        
        # I have to check if different nodes have the same input set in terms of colors
        # Find unique color count vectors
        H = list(set(I))
        
        # Assign new colors based on the index in H
        # Two nodes with the same colors in input will have the same new color assigned
        new_c = {}
        for node, Ii in zip(G.nodes(), I):
            new_c[node] = H.index(Ii)# + next_color
        
        # Update Nj
        new_Nj = len(H)
        
        # Continue the procedure until the set of colors does not change
        if new_Nj == Nj:
            break
        
        c = new_c
        Nj = new_Nj
    
    return c



def graph_to_bipartite(G):
    """
    Transform a graph into a bipartite graph.

    Parameters:
        G: A NetworkX graph
    Return:
        A NetworkX Graph object representing the bipartite graph
    """
    B = nx.Graph()
    
    # Add nodes from the original graph to the first layer
    B.add_nodes_from(G.nodes(), bipartite=0)

    N = len(G.nodes())
    
    # Add nodes for the edges and connect to the vertices
    for i, edge in enumerate(G.edges()):
        edge_node = i + N
        B.add_node(edge_node, bipartite=1)
        B.add_edge(edge_node, edge[0])
        B.add_edge(edge_node, edge[1])
    
    return B



def kameicock_coloring_multi(G, initial_types=None):
    """
    Color a MultiGraph using Kamei & Cock's algorithm, accounting for multiple edges between nodes.

    Parameters:
        G: A NetworkX MultiGraph
        initial_types: A dictionary of {node: initial_color}
    Return:
        A dictionary of {node: color} for all nodes
    """
    if initial_types is None:
        initial_types = {}

    # Initialize color mapping
    c = initial_types.copy()
    next_color = max(initial_types.values(), default=0) + 1

    for node in G.nodes():
        if node not in c:
            c[node] = next_color

    Nj = max(c.values())

    while True:
        I = []

        for node in G.nodes():
            color_count = [0] * (Nj + 1)

            for neighbor in G.neighbors(node):
                edge_count = G.number_of_edges(node, neighbor)
                neighbor_color = c[neighbor]
                if neighbor_color <= Nj:
                    color_count[neighbor_color] += edge_count

            I.append(tuple(color_count))

        H = list(set(I))
        new_c = {node: H.index(Ii) for node, Ii in zip(G.nodes(), I)}
        new_Nj = len(H)

        if new_Nj == Nj:
            break

        c = new_c
        Nj = new_Nj

    return c

# Function used in MAG_fibration_comparison notebook to find differences in fibres according to different hypergraph representations
def find_hyperedges_containing_all_nodes(hyperedges, target_nodes):
    """
    Find all hyperedges that contain all nodes in a given set.

    Parameters:
        hyperedges: List of hyperedges (each hyperedge is a list of nodes)
        target_nodes: Iterable of nodes that must all be present in the hyperedge

    Returns:
        List of hyperedges containing all target nodes
    """
    target_set = set(target_nodes)
    return [he for he in hyperedges if target_set.issubset(he)]


def normalize_sequence(lst):
    """
    Normalizes a list of lists to ensure all integers from 0 to max appear consecutively.

    Parameters:
        lst: List of lists.

    Returns:
        list of lists: A normalized list of lists with no gaps in the sequence.
    """
    # Flatten the list of lists and find unique integers
    unique_integers = sorted(set(num for sublist in lst for num in sublist))

    # Create a mapping of old elements to new consecutive integers
    mapping = {old: new for new, old in enumerate(unique_integers)}

    # Apply the mapping to normalize the list of lists
    normalized_lst = [[mapping[num] for num in sublist] for sublist in lst]

    return normalized_lst, mapping


## To compute the fibre partition in a unique function for a list of hypergraphs

def hypergraph_coloring_list(hypergraphs_list):

    """
    Generate all possible cluster given the list of hypergraphs.
    
    Parameters:
        hypergraphs_list: A list of list with all the possible hypergraphs
    Return:
        partition_result: An array having 
        0. e possible hypergraphs in input
        1. the clustering of nodes and hyperdges obtained applying the fibration algorithm to its bipartite representation.
    """

    # Create the list of all possible combinations of hyperdges
    h = hypergraphs_list

    kc_res = []
    kc_dict_list = []
    color_dict_list = []

    for i in range(len(h)):

        # Construct the ith hypergraph 
        hg_iter = Hypergraph(h[i])

        # Get its bipartite projection
        bhg_iter_tot = bipartite_projection(hg_iter)
        bhg_iter = bhg_iter_tot[0]
        bhg_dict = bhg_iter_tot[1]
        # Get the coloring of the bipartite graph
        color_dict_iter = bipartite.color(bhg_iter)

        # Apply the algorithm
        res = kameicock_coloring(bhg_iter, color_dict_iter)

        kc_res.append(res)
        kc_dict_list.append(color_dict_iter)
        color_dict_list.append(bhg_dict)

    partition_result = np.array([h, kc_res, color_dict_list], dtype=object).T    

    return partition_result