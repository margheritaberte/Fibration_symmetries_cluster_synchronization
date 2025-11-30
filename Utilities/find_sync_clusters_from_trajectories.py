import numpy as np
from itertools import combinations
from tqdm import tqdm
from numba import njit, prange
import networkx as nx

@njit(parallel=True)
def quick_phase_alignment_check_numba(sampled, phase_tol):

    """ Check if all rows in sampled have phases aligned within phase_tol. 
    Parameters:
        sampled: 2D array of shape (n_times, n_nodes)
        phase_tol: tolerance for phase difference
    Returns:
        bool: True if all rows have phases aligned within phase_tol, False otherwise
    """

    for t in prange(sampled.shape[0]):
        row = sampled[t]
        # Check if the max-min difference in the row exceeds phase_tol
        if np.max(row) - np.min(row) >= phase_tol:
            return False
    return True

def quick_phase_alignment_check(theta_subset, sample_timesteps, phase_tol=0.05):
    """
    Check if all rows in the subset of theta values have phases aligned within phase_tol.

    Parameters:
        theta_subset: 2D array of shape (n_times, n_nodes)
        sample_timesteps: array-like of time indices to sample
        phase_tol: tolerance for phase difference

    Returns:
        bool: True if all sampled rows have phases aligned within phase_tol, False otherwise
    """
    sampled = theta_subset[sample_timesteps, :]
    return quick_phase_alignment_check_numba(sampled, phase_tol)

def kuramoto_order_parameter_over_time(theta_subset):
    """
    Compute the Kuramoto order parameter over time for a subset of theta values.

    Parameters:
        theta_subset: 2D array of shape (n_times, n_nodes)

    Returns:
        1D array of Kuramoto order parameter values over time
    """
    return np.abs(np.sum(np.exp(1j * theta_subset), axis=1)) / theta_subset.shape[1]


def find_synchronized_clusters_from_pairs(theta_history, nodes_by_degree, 
                                          tol=1e-6, phase_tol=0.05, n_check_times=5):
    """
    Find synchronized clusters from pairs of nodes based on their phase history.

    Parameters:
        theta_history: 2D array of shape (T, N) representing phase history over time
        nodes_by_degree: dictionary mapping degree sequences to lists of nodes
        tol: tolerance for Kuramoto order parameter deviation from 1
        phase_tol: tolerance for phase difference in quick alignment check
        n_check_times: number of time points to sample for quick alignment check

    Returns:
        List of synchronized clusters as list of lists (each cluster is a list of node indices)
    """
    T, N = theta_history.shape
    synchronized_clusters = []

    # Choose random time samples for quick phase alignment check
    time_samples = np.random.choice(T, size=n_check_times, replace=False)

    for degree_seq, nodes in tqdm(nodes_by_degree.items()):

        # Skip degree sequences with just one node
        if len(nodes) < 2:
            continue

        print(f'Degree sequence: {degree_seq} ({len(nodes)} nodes)')

        # Initialize an empty graph to track synchronized pairs
        sync_graph = nx.Graph()
        # Add nodes to the graph
        sync_graph.add_nodes_from(nodes)

        for i, j in combinations(nodes, 2):
            sub_theta = theta_history[:, [i, j]]

            # Check phase alignment before computing order parameter
            if not quick_phase_alignment_check(sub_theta, time_samples, phase_tol):
                continue

            # Compute Kuramoto order parameter over time
            R = kuramoto_order_parameter_over_time(sub_theta)

            # If R is close to 1 within tolerance, add an edge
            if np.all(np.abs(R - 1) < tol):
                sync_graph.add_edge(i, j)

        # Extract non-trivial connected components as synchronized clusters 
        for component in nx.connected_components(sync_graph):
            if len(component) > 1:
                synchronized_clusters.append(sorted(list(component)))

    return synchronized_clusters