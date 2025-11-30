import numpy as np
from copy import deepcopy

from collections import defaultdict
import random
import math
from math import gcd
from itertools import cycle, islice

from functools import reduce

import networkx as nx
from networkx.algorithms import bipartite

import hypergraphx as hx
from hypergraphx.core.hypergraph import Hypergraph

# Custom imports
from utilities_fibration import *
from draw_hypergraph_col import *

def is_connected_hypergraph(hyperedges, all_nodes=None):
    """
    Check if the hypergraph is connected.

    Parameters:
        hyperedges: list of hyperedges (as list of lists)
        all_nodes: optional set of all nodes in the hypergraph

    Returns:
        bool: True if the hypergraph is connected, False otherwise
    """

    # Initialize an undirected graph
    G = nx.Graph()
    # Add nodes and edges based on hyperedges
    if all_nodes:
        G.add_nodes_from(all_nodes)
    # Take all the hyperedges and consider the projection to a simple graph
    for hedge in hyperedges:
        for i in range(len(hedge)):
            for j in range(i + 1, len(hedge)):
                G.add_edge(hedge[i], hedge[j])
    return nx.is_connected(G)

def remove_color_and_check_clusters_iteratively(
    hyperedges,
    color_dict,
    node_map,
    c_original,
    color_order=None,
    respect_protected_hyperedges=False,
    protected_hyperedges=None
):
    """
    Remove colors from the hypergraph iteratively and check cluster consistency.

    Parameters:
        hyperedges: list of hyperedges as list of lists of integers
        color_dict: dictionary mapping hyperedges to colors
        node_map: dictionary mapping node names to nodes as integers
        c_original: original clusters as list of lists
        color_order: order in which to remove colors (optional)
        respect_protected_hyperedges: whether to respect protected hyperedges
        protected_hyperedges: list of protected hyperedges (optional)

    Returns:
        Tuple containing updated hyperedges, color dictionary, node map, and clusters
    """
    # Map hyperedge name -> color
    hedge_colors = {k: v for k, v in color_dict.items() if k.startswith('E')}
    
    # Map hyperedge as list of sets of nodes (so no problem for permutation)
    color_to_hedge_sets = defaultdict(list)
    hedge_name_to_nodeset = {k: frozenset(node_map[k]) for k in hedge_colors}

    # Build mapping from color to hyperedge sets
    for hedge_name, color in hedge_colors.items():
        hedge_nodeset = hedge_name_to_nodeset[hedge_name]
        color_to_hedge_sets[color].append(hedge_nodeset)

    if color_order is None:
        # Sort colors by size of hyperedges associated (largest first)
        color_order = sorted(
            color_to_hedge_sets.keys(),
            key=lambda c: max(len(h) for h in color_to_hedge_sets[c])
        )

    # Check if protected_hyperedges provided
    if protected_hyperedges is None:
        protected_hyperedges = set()
    else:
        # Convert to set of frozensets for comparison
        protected_hyperedges = set(frozenset(h) for h in protected_hyperedges)

    all_nodes = set(node for cluster in c_original for node in cluster)
    current_hyperedges = deepcopy(hyperedges)
    current_node_map = deepcopy(node_map)
    current_clusters = None

    for color in color_order:
        candidate_hedge_sets = set(color_to_hedge_sets[color])

        if respect_protected_hyperedges:
            # Exclude protected hyperedges from removal
            hedge_sets_to_remove = candidate_hedge_sets - protected_hyperedges
        else:
            hedge_sets_to_remove = candidate_hedge_sets

        if hedge_sets_to_remove != set():

            # Filter node_map entries by matching hyperedge set
            new_node_map = {
                name: nodes for name, nodes in current_node_map.items()
                if not (name.startswith('E') and frozenset(nodes) in hedge_sets_to_remove)
            }

            new_hyperedges = [list(new_node_map[k]) for k in new_node_map if k.startswith('E')]

            # Check if hypergraph is still connected
            if is_connected_hypergraph(new_hyperedges, all_nodes):
                print(f"Removed hyperedges of original color {color}. Hypergraph is still connected.")

                # Check cluster consistency after removal
                success, updated_hyperedges, updated_color_dict, updated_node_map, clusters = check_cluster(new_hyperedges, c_original)

                if success:
                    print(f"Clusters are consistent after removing color {color}.")
                    current_hyperedges = updated_hyperedges
                    current_node_map = updated_node_map
                    current_clusters = clusters
                else:
                    print(f"Cluster check failed after removing color {color}. Rolling back.")
            else:
                print(f"Removal of color {color} disconnects the hypergraph. Rolling back.")

    return current_hyperedges, color_dict, current_node_map, current_clusters


def optimize_color_removal(hyperedges, color_dict, node_map, c_original, max_permutations=100,
    respect_protected_hyperedges=False, protected_hyperedges=None):

    """
    Optimize the removal of colors from the hypergraph to maintain cluster consistency.

    Parameters:
        hyperedges: list of hyperedges
        color_dict: dictionary mapping hyperedges to colors
        node_map: dictionary mapping node names to nodes
        c_original: original clusters
        max_permutations: maximum number of permutations to try
        respect_protected_hyperedges: whether to respect protected hyperedges
        protected_hyperedges: list of protected hyperedges (optional)

    Returns:
        Tuple containing updated hyperedges, color dictionary, node map, and clusters
    """

    fl_resp = respect_protected_hyperedges
    prot_list = protected_hyperedges

    # Flatten all colors ignoring order grouping for random shuffle each time
    all_colors = list({v for k, v in color_dict.items() if k.startswith('E')})

    permutations_tried = 0
    best_result = None
    best_num_edges = float('inf')

    while permutations_tried < max_permutations:
        # Shuffle the color order freshly for each iteration
        random.shuffle(all_colors)

        result = remove_color_and_check_clusters_iteratively(
            hyperedges, color_dict, node_map, c_original, all_colors, fl_resp, prot_list
        )
        remaining_edges = len(result[0])
        print(f'len: {remaining_edges}')
        if remaining_edges < best_num_edges:
            best_result = result
            best_num_edges = remaining_edges

        permutations_tried += 1

        print(f"Tried {permutations_tried} permutations.\n\n*************************")
    return best_result


def normalize_partition(partition):
    """
    Convert partition to comparable frozenset format.
    Parameters:
        partition: list of clusters (each cluster is a list of nodes)
    """
    return frozenset(frozenset(sorted(group)) for group in partition)

def compute_fibers(hyperedges):
    """
    Compute fibers (clusters) in the hypergraph based on hypergraph coloring.
    
    Parameters:
        hyperedges: list of hyperedges (each hyperedge is a list of nodes)
    
    Returns:
        Tuple containing:
        - updated hyperedges
        - updated color dictionary
        - updated node map
        - clusters
    """
    # Compute hypergraph coloring
    result = hypergraph_coloring_list([hyperedges])
    edge_name_dict = result[0][2]
    edge_col_dict = result[0][1]
    
    node_cluster_hg = {k: v for k, v in edge_col_dict.items() if k.startswith('N')}
    cl_hg = []
    for val in set(node_cluster_hg.values()):
        cl_hg.append([k for k, v in node_cluster_hg.items() if v == val])
    
    clusters = [[edge_name_dict[item] for item in sublist] for sublist in cl_hg]
    return result[0][0], edge_col_dict, edge_name_dict, clusters

def check_cluster(new_hyperedges, c_original):
    """
    Check if the new hyperedges produce clusters compatible with the original clusters.

    Parameters:
        new_hyperedges: list of new hyperedges as list of lists
        c_original: original clusters as list of lists

    Returns:
        Tuple 
        containing:
            bool indicating if clusters are compatible
            updated hyperedges
            updated color dictionary
            updated node map
            clusters as list of lists
    """
    # Compute hypergraph coloring on new hyperedges
    hyperedges_updated, edge_col_dict_updated, edge_name_updated, clusters = compute_fibers(new_hyperedges)
    

    # Convert clusters to sets of frozensets for comparison
    set_pruned = set(frozenset(inner) for inner in c_original)
    set_original = set(frozenset(inner) for inner in clusters)

    # Compare the sets of clusters
    if set_pruned == set_original:
        return True, hyperedges_updated, edge_col_dict_updated, edge_name_updated, clusters
    else:
        ###print('New clusters not compatible')
        return False, hyperedges_updated, edge_col_dict_updated, edge_name_updated, clusters


def get_node_neighbors(node, hyperedges):
    """
    Get all nodes that are connected to the given node through hyperedges.
    Parameters:
        node: The node for which neighbors are to be found
        hyperedges: list of hyperedges (each hyperedge is a list of nodes)
    Returns:
        set of neighboring nodes
    """
    neighbors = set()
    for hedge in hyperedges:
        if node in hedge:
            # Add all nodes in the hyperedge
            neighbors.update(hedge) # Add all nodes in the hyperedge
    neighbors.discard(node)  # Remove the node itself
    return neighbors

def get_node_hyperedge_patterns(node, hyperedges):
    """
    Get all hyperedge patterns (sorted tuples) that include the given node.
    Parameters:
        node: The node for which hyperedge patterns are to be found
        hyperedges: list of hyperedges (each hyperedge is a list of nodes)
    """
    patterns = set()
    for hedge in hyperedges:
        if node in hedge:
            # Create tuple of nodes in the hyperedge without the node itself
            pattern = tuple(sorted(n for n in hedge if n != node))
            if pattern:  # Only add non-empty patterns
                patterns.add(pattern)
    return patterns

def should_split_cluster(cluster, target_clusters):
    """
    Check if a cluster should be split by verifying that its nodes
    are not supposed to be part of a larger target cluster.
    
    Args:
        cluster: Current cluster (frozenset of nodes)
        target_clusters: List of target clusters (frozensets)
    
    Returns:
        bool: True if cluster should be split, False otherwise
    """
    cluster_nodes = set(cluster)
    
    # Check if any target cluster contains all nodes from this cluster or overlaps with more than one node
    # In this case it should not be split but at most merged later
    for target_cluster in target_clusters:
        # If the target cluster fully contains the current cluster or if there's partial overlap with more than one node
        if cluster_nodes.issubset(set(target_cluster)) or (len(cluster_nodes.intersection(set(target_cluster))) > 0 and len(target_cluster) > 1):
            print(target_cluster)
            print(len(cluster_nodes.intersection(set(target_cluster))))
            return False
    
    return True

def split_clusters(hyperedges, clusters_to_split, all_nodes, step, allow_multiplicity=True):
    """
    Add hyperedges to split clusters that shouldn't be together.
    Connect each node in unwanted clusters to different external nodes.
    Parameters:
        hyperedges: list of hyperedges (each hyperedge is a list of nodes)
        clusters_to_split: list of clusters to split (each cluster is a frozenset of nodes)
        all_nodes: set of all nodes in the hypergraph
        step: current step number (used for cycling through nodes)
        allow_multiplicity: whether to allow multiple copies of same hyperedge
    Returns:
        int: Number of edges added
    """
    #external_nodes = list(all_nodes - set().union(*clusters_to_split))
    edges_added = 0
    
    for cluster in clusters_to_split:
        # Find all nodes not in this cluster
        external_nodes = list(all_nodes - set().union(cluster))
        cluster_list = list(cluster)

        # Start from position (step % cluster_size) for changing across iterations
        for i, node in enumerate(cluster_list[step % len(cluster_list):]):
            edges_add = 0
            
            # Try to connect this node to some external node
            for k in range(len(external_nodes)):
                new_edge = [node, external_nodes[k]]
                
                normalized_hyperedges = {normalize_hyperedge(h) for h in hyperedges}
                if not hyperedge_exists(normalized_hyperedges, new_edge):
                    hyperedges.append(new_edge)  # Add the edge
                    edges_added += 1
                    edges_add = 1
                    break  # Found one, move to next cluster node
                else:
                    continue  # This edge exists, try next external node
            
            # If no unique edge found and multiplicity allowed
            if edges_add == 0 and allow_multiplicity:
                if i < len(external_nodes):
                    new_edge = [node, external_nodes[i]]
                hyperedges.append(new_edge)  # Add duplicate edge
                edges_added += 1
                print(f"    Added duplicate edge: {new_edge}")
            elif edges_add == 0:
                print(f"    No edges added")
    
    return edges_added

def normalize_hyperedge(hedge):
    """Normalize a hyperedge to a sorted tuple for comparison.
    Parameters:
        hedge: list of nodes in the hyperedge
    Returns:
        tuple of sorted nodes representing the hyperedge
    """
    return tuple(sorted(hedge))

def hyperedge_exists(normalized_hyperedges, new_hedge):
    """Check if a hyperedge already exists in the list.
    Parameters:
        normalized_hyperedges: set of normalized existing hyperedges (each hyperedge is a tuple of nodes)
        new_hedge: list of nodes in the new hyperedge to check
    Returns:
        bool: True if the hyperedge exists, False otherwise
    """
    normalized_new = normalize_hyperedge(new_hedge)
    return normalized_new in normalized_hyperedges

def merge_clusters(hyperedges, clusters_to_merge, allow_multiplicity=True):
    """
    Add hyperedges to merge clusters that should be together.
    Make nodes in the same target cluster have the same connectivity pattern.
    Parameters:
        hyperedges: list of hyperedges (each hyperedge is a list of nodes)
        clusters_to_merge: list of clusters to merge (each cluster is a frozenset of nodes)
        allow_multiplicity: whether to allow multiple copies of same hyperedge
    Returns:
        int: Number of edges added
    """
    edges_added = 0
    
    for cluster in clusters_to_merge:
        print(f"  Merging cluster: {sorted(cluster)}")
        cluster_list = list(cluster)
        
        if len(cluster_list) < 2:
            continue
        
        # Equalize external patterns
        # Get hyperedge patterns for each node in the cluster
        node_patterns = {}
        for node in cluster_list:
            node_patterns[node] = get_node_hyperedge_patterns(node, hyperedges)
        
        # Find all unique patterns across the cluster
        all_patterns = set()
        for patterns in node_patterns.values():
            all_patterns.update(patterns)
        
        # Remove patterns that only contain cluster members (internal patterns)
        external_patterns = set()
        cluster_set = set(cluster_list)
        for pattern in all_patterns:
            #if not set(pattern).issubset(cluster_set):
            external_patterns.add(pattern)
        
        print(f"    External patterns to match: {sorted(external_patterns)}")
        
        # Make all nodes in cluster have the same external patterns
        for node in cluster_list:
            current_patterns = node_patterns[node]
            missing_patterns = external_patterns - current_patterns
            
            for pattern in missing_patterns:
                # Create new hyperedge with the node and the pattern
                new_hyperedge = [node] + list(pattern)
                
                # Check if this hyperedge already exists
                normalized_hyperedges = {normalize_hyperedge(h) for h in hyperedges}
                if not hyperedge_exists(normalized_hyperedges, new_hyperedge) and len(new_hyperedge) == len(set(new_hyperedge)):
                    hyperedges.append(new_hyperedge)
                    edges_added += 1
                    print(f"    Added hyperedge: {new_hyperedge}")
                elif allow_multiplicity:
                    hyperedges.append(new_hyperedge)
                    edges_added += 1
                    print(f"    Added duplicate hyperedge: {new_hyperedge}")
                else:
                    print(f"    Skipped duplicate hyperedge: {new_hyperedge}")
        
        # If no external patterns exist, try pairwise connections as fallback
        if not external_patterns:
            print(f"    No external patterns found, using pairwise connections")
            # Get current neighbors for each node
            node_neighbors = {}
            for node in cluster_list:
                node_neighbors[node] = get_node_neighbors(node, hyperedges)
            
            # Find all unique neighbors across the cluster
            all_neighbors = set()
            for neighbors in node_neighbors.values():
                all_neighbors.update(neighbors)
            
            # Remove cluster members from neighbors (internal connections)
            all_neighbors = all_neighbors - cluster_set
            
            # Make all nodes in cluster have the same external connections
            for node in cluster_list:
                current_neighbors = node_neighbors[node] - cluster_set
                missing_neighbors = all_neighbors - current_neighbors
                
                # Add edges to missing neighbors
                for neighbor in missing_neighbors:
                    new_edge = [node, neighbor]
                    normalized_hyperedges = {normalize_hyperedge(h) for h in hyperedges}
                    if not hyperedge_exists(normalized_hyperedges, new_edge):
                        hyperedges.append(new_edge)
                        edges_added += 1
                        print(f"    Added edge: {new_edge}")
                    elif allow_multiplicity:
                        hyperedges.append(new_edge)
                        edges_added += 1
                        print(f"    Added duplicate edge: {new_edge}")
                    else:
                        print(f"    Skipped duplicate edge: {new_edge}")
    
    return edges_added

def refine_hypergraph(G, C, max_steps=20, allow_multiplicity=True):
    """
    Refine hypergraph G to achieve target clustering C.
    
    Parameters:
        G: Initial hypergraph as list of lists
        C: Target clusters as list of lists
        max_steps: Maximum number of refinement steps
        allow_multiplicity: Whether to allow multiple copies of same hyperedge
    
    Returns:
        Refined hypergraph as list of lists
    """
    print("Starting hypergraph refinement...")
    print(f"Initial hypergraph: {G}")
    print(f"Target clusters: {sorted([sorted(c) for c in C])}")
    print(f"Allow multiplicity: {allow_multiplicity}")
    
    G = deepcopy(G)
    C_norm = normalize_partition(C)
    C_norm_list = list(C_norm)  # Convert to list for easier access
    all_nodes = {node for edge in G for node in edge}
    
    for step in range(1, max_steps + 1):
        print(f"\n=== Step {step} ===")
        
        # Compute current fiber partition
        P = compute_fibers(G)[3]  # Get clusters from the result
        P_norm = normalize_partition(P)
        
        print(f"Current partition: {sorted([sorted(cluster) for cluster in P])}")
        print(f"Current hyperedges: {G}")
        
        # Check if target is reached
        if P_norm == C_norm:
            print("Target partition reached!")
            break
        
        # Find clusters that need to be merged (in C but not in P)
        clusters_to_merge = []
        for c_cluster in C_norm:
            if c_cluster not in P_norm and len(c_cluster) > 1:
                clusters_to_merge.append(c_cluster)

        # Find clusters that need to be split (in P but not in C)
        # Consider them only if they're not supposed to be part of a larger target cluster
        clusters_to_split = []
        for p_cluster in P_norm:
            if p_cluster not in C_norm and len(p_cluster) > 1:
                if should_split_cluster(p_cluster, C_norm_list):
                    clusters_to_split.append(p_cluster)
                else:
                    print(f"Skipping split for cluster {sorted(p_cluster)} - nodes belong to larger target cluster")
        
        print(f"Need to split: {[sorted(c) for c in clusters_to_split]}")
        print(f"Need to merge: {[sorted(c) for c in clusters_to_merge]}")
        
        # Merge desired clusters
        merge_edges = 0
        if clusters_to_merge:
            merge_edges = merge_clusters(G, clusters_to_merge, allow_multiplicity)
            
        # Split unwanted clusters
        split_edges = 0
        if clusters_to_split:
            split_edges = split_clusters(G, clusters_to_split, all_nodes, step, allow_multiplicity)
        
        total_edges_added = split_edges + merge_edges
        print(f"Total edges added this step: {total_edges_added}")
        
        # If no edges were added, we might be stuck
        if total_edges_added == 0:
            print("No new edges added. Checking if we're stuck...")
            if clusters_to_split or clusters_to_merge:
                print("WARNING: Clusters need changes but no edges were added. Possible convergence issue.")
                # Try: add a direct edge to force progress
                cluster = None
                if clusters_to_merge:
                    cluster = list(clusters_to_merge)[0]
                elif clusters_to_split:
                    cluster = list(clusters_to_split)[0]
                if cluster:
                    cluster_list = list(cluster)
                    if len(cluster_list) >= 2:

                        for i in range(len(cluster_list)):
                            for j in range(i+1, len(cluster_list)):
                                new_edge = [cluster_list[i], cluster_list[j]]

                                normalized_hyperedges = {normalize_hyperedge(h) for h in G}
                                if not hyperedge_exists(normalized_hyperedges, new_edge):
                                    G.append(new_edge)
                                    print(f"    Fallback: added direct connection {new_edge}")
                                    total_edges_added += 1
                                    break
                                elif allow_multiplicity:
                                    G.append(new_edge)
                                    print(f"    Fallback: added duplicate direct connection {new_edge}")
                                    total_edges_added += 1
                                    break
                            if total_edges_added > 0:
                                break
            
            if total_edges_added == 0:
                print("Still no edges added. Algorithm may be stuck.")
                break
    
    else:
        print(f"Maximum steps ({max_steps}) reached without convergence.")
    
    return G

def lcm(numbers):
    """Compute the least common multiple of a list of numbers.
    Parameters:
        numbers: list of integers
    Returns:
        int: least common multiple of the input numbers
    """
    # Python 3.9+ has built-in LCM for multiple numbers
    if hasattr(math, 'lcm'):
        return math.lcm(*numbers)
    
    # For older Python versions
    return _manual_lcm(numbers)


def lcm_pair(a, b):
    """Compute the least common multiple of two numbers.
    Parameters:
        a: integer
        b: integer
    Returns:
        int: least common multiple of a and b
    """
    return abs(a * b) // gcd(a, b)

def _manual_lcm(numbers):
    """Compute the least common multiple of a list of numbers.
    Parameters:
        numbers: list of integers
    Returns:
        int: least common multiple of the input numbers
    """
    # Apply function to all numbers together
    return reduce(lcm_pair, numbers, 1)

def add_structured_redundant_hyperedges_preserving_fibers(
    hyperedges,
    color_dict,
    node_map,
    c_original,
    max_new_hyperedges=10,
    max_order=None,
    max_trials=100
):
    """
    Add structured redundant hyperedges while preserving the original cluster structure.

    Parameters:
        hyperedges: list of lists of node labels
        color_dict: dictionary of node/edge colors
        node_map: dict mapping edge/node names to sets of nodes
        c_original: list of clusters (original partition to preserve)
        max_new_hyperedges: total number of redundant hyperedges to add
        max_order: maximum allowed hyperedge size (defaults to largest existing edge)
        max_trials: maximum number of batch trials before stopping

    Returns:
        updated_hyperedges, updated_color_dict, updated_node_map, updated_clusters
    """
    # Determine maximum order if not provided
    if max_order is None:
        max_order = max(len(h) for h in hyperedges)

    all_nodes = sorted(set(node for cluster in c_original for node in cluster))
    # Track existing hyperedges as frozensets to detect duplicates
    existing_hedges = set(frozenset(h) for h in hyperedges)

    # Initialize current structures
    current_hyperedges = list(hyperedges)
    current_node_map = dict(node_map)
    new_color_dict = dict(color_dict)
    added_total = 0
    edge_counter = 0
    # Create an infinite cycle through possible hyperedge orders
    order_cycle = cycle(range(2, max_order + 1))

    trial = 0
    # Keep trying until we reach max_trials or add enough hyperedges
    while trial < max_trials and added_total < max_new_hyperedges:
        # Get the next hyperedge size to try
        order = next(order_cycle)
        trial += 1

        # Skip if we don't have enough clusters for this order
        # (e.g. can't create an order-5 hyperedge from only 3 clusters)
        if order > len(c_original):
            continue  # not enough clusters
        
        # Randomly select a number of clusters equal to the order
        chosen_clusters = random.sample(c_original, order)

        # Try up to 3 times to select nodes from clusters that yield valid new hyperedges
        for _ in range(3):
            try:
                # Get the sizes of the chosen clusters
                sizes = [len(cluster) for cluster in chosen_clusters]
                # Compute the least common multiple of the cluster sizes to determine how many hyperedges to create
                lcm_size = lcm(sizes)

                # For each cluster, create a cycled list of its nodes repeated to lcm_size.                
                node_lists = [
                    list(islice(cycle(random.sample(cluster, len(cluster))), lcm_size))
                    for cluster in chosen_clusters
                ]

                # Zip the node lists together to create hyperedges
                # Each hyperedge takes one node from each cluster
                candidate_hyperedges = [list(nodes) for nodes in zip(*node_lists)]

            except ValueError:
                continue  # one of the clusters too small, skip this attempt

            # Check if any of the new hyperedges already exists
            if any(frozenset(h) in existing_hedges for h in candidate_hyperedges):
                continue  # retry node sampling

            # Tentatively add these hyperedges
            new_node_map = dict(current_node_map)
            new_hyperedges = list(current_hyperedges)

            # Add all candidate hyperedges to the temporary structures
            for hedge in candidate_hyperedges:
                edge_name = f"E_new_{edge_counter}"
                new_node_map[edge_name] = hedge
                new_hyperedges.append(hedge)
                edge_counter += 1

            # Validate that adding these hyperedges preserves the cluster structure
            success, hyperedges_updated, updated_color_dict, updated_node_map, clusters = check_cluster(new_hyperedges, c_original)

            if success:
                # Check if adding this batch would exceed our limit
                if added_total + len(candidate_hyperedges) <= max_new_hyperedges:
                    print(f"Added {len(candidate_hyperedges)} hyperedges of order {order} connecting clusters {[c_original.index(cl) for cl in chosen_clusters]}.")

                    # Mark these hyperedges as existing (to avoid duplicates later)
                    for hedge in candidate_hyperedges:
                        existing_hedges.add(frozenset(hedge))

                    # Commit the validated changes to our working structures    
                    current_hyperedges = new_hyperedges
                    current_node_map = updated_node_map
                    new_color_dict = updated_color_dict
                    added_total += len(candidate_hyperedges)
                else:
                    print(f"Batch of {len(candidate_hyperedges)} hyperedges exceeds max limit. Discarded.")
                break  # don't retry this cluster selection
        else:
            # This executes if the for loop completes without breaking
            print(f"Failed to add valid batch of order {order} after 3 node selections.")

    return current_hyperedges, new_color_dict, current_node_map, c_original
