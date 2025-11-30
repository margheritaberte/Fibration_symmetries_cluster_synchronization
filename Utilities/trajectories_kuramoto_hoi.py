import numpy as np
from numpy import linalg
from typing import List, Tuple, Optional, Union
from scipy.integrate import solve_ivp

class KuramotoHypergraph:
    def __init__(
        self,
        hyperedges: List[List[int]],
        omega: np.ndarray,
        sigma: List[float],
        alpha: List[float]
    ):
        """
        Initialize the Kuramoto model with hypergraph interactions.
        
        Parameters:
        hyperedges : List[List[int]]
            List of hyperedges. Each hyperedge is a list of node indices.
        omega : np.ndarray
            Natural frequencies of oscillators (shape: N)
        sigma : List[float]
            List of coupling strengths. sigma[i] corresponds to coupling strength
            for (i+2)-body interactions
        alpha : List[float]
            List of phase shifts. alpha[i] corresponds to phase shift
            for (i+2)-body interactions
        """

        self.N = len(omega)  # Number of oscillators
        self.omega = omega
        self.sigma = sigma
        self.alpha = alpha
        
        # Organize hyperedges by their order
        max_order = len(sigma) + 1  # Maximum order of interactions
        self.edges_by_order = [[] for _ in range(max_order + 1)]
        for edge in hyperedges:
            order = len(edge)
            if order >= 2 and order <= max_order:
                self.edges_by_order[order].append(edge)
        
        
    def derivatives(self, t: float, theta: np.ndarray) -> np.ndarray:
        """
        Compute the derivatives of phases according to the Kuramoto model.
        Note: Parameter order in the form (t, y) as required by solve_ivp
        
        Parameters:
        t : float
            Current time 
        theta : np.ndarray
            Current phases of oscillators
            
        Returns:
        np.ndarray
            Derivatives of phases
        """
        # Initialize derivatives with natural frequencies
        if self.alpha == 0:
            alpha = np.zeros(len(self.sigma))
        else:
            alpha = self.alpha

        dtheta = self.omega.copy()
        
        # Process all orders of interactions
        for order in range(2, len(self.edges_by_order)):
            
            sigma_idx = order - 2  # Index in sigma list
            if sigma_idx < len(self.sigma):  # Check if we have coupling strength for this order
                for edge in self.edges_by_order[order]:
                    # For each node in the hyperedge
                    for i in edge:
                        # Get other nodes in the hyperedge
                        others = [j for j in edge if j != i]                    

                        dtheta[i] += (self.sigma[sigma_idx]) * np.sin(sum(theta[j] for j in others) - (order-1)*theta[i] - alpha[sigma_idx])
                        
        #print(theta)
        return dtheta
    

    def simulate(self, T: int, dt: float, initial_conditions: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the Kuramoto model using scipy's solve_ivp.
        
        Parameters:
        T : int
            Number of time steps
        dt : float
            Time step size
        initial_conditions : np.ndarray, optional
            Initial phases (if None, random phases are used)
            
        Returns:
        Tuple[np.ndarray, np.ndarray]
            Time points and phase evolution
        """
        # Initialize phases
        if initial_conditions is None:
            theta0 = 2 * np.pi * np.random.random(self.N)
        else:
            theta0 = initial_conditions
            
        # Create time evaluation points
        t_eval = np.arange(0, T * dt, dt)
        
        # Solve using solve_ivp
        solution = solve_ivp(
            self.derivatives,
            t_span=[0, T * dt],
            y0=theta0,
            t_eval=t_eval,
            method='LSODA' ,
            rtol=1e-9, atol=1e-9
        )
        
        return solution.t, solution.y.T

    def compute_order_parameter(theta: np.ndarray) -> float:
        """Compute the Kuramoto order parameter r for a set of phases.
        Parameters:
            theta : np.ndarray
                Phases of all oscillators
        Returns:
            float
                Kuramoto order parameter r
        """
        return np.abs(np.mean(np.exp(1j * theta)))

    def compute_cluster_order_parameters(theta: np.ndarray, clusters: List[Union[List[int], int]]) -> List[float]:
        """
        Compute order parameters for specified clusters.
        
        Parameters:
        theta : np.ndarray
            Phases of all oscillators
        clusters : List[Union[List[int], int]]
            Specification of node clusters. Each element can be either:
            - A list of node indices forming a cluster
            - A single integer representing a node in its own cluster
            
        Returns:
        List[float]
            List of order parameters, one for each cluster
        """
        # Convert single integers to lists
        cluster_lists = [[x] if isinstance(x, int) else x for x in clusters]
        
        # Compute order parameter for each cluster
        return [KuramotoHypergraph.compute_order_parameter(theta[cluster]) 
                for cluster in cluster_lists]
