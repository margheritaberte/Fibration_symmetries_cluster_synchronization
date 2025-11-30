# Fibration Symmetries and Cluster Synchronization in Multi-Body Systems

This repository contains the code used for the analysis presented in the paper:  
_Bertè, Margherita, Tommaso Gili ["Fibration Symmetries and Cluster Synchronization in Multi-Body Systems."](https://arxiv.org/abs/2510.11207)(2025)._

## Contents
- 'Utilities/': folder containing python scripts with functions used in the notebooks.
In particular the file **utilities_fibration.py** includes all the functions to compute the fibre partition (aka coarsest equitable partition or minimal balanced coloring) of nodes for graphs, multigraphs, hypergraphs.
- 'Data/': contains the MAG-10 dataset used as example in the Notebook **MAG_fibration_comparison.ipynb** (preprocessed and available at (https://github.com/TheoryInPractice/overlapping-ecc/tree/master/data/MAG-10)).
- 'Examples_computing_fibres/': folder containing Jupyter Notebooks with examples of how to compute the fibre partition for hypergraphs and multigraphs.
- 'Comparing_fibrations/': folder containing Jupyter Notebooks with examples of how to compare the fibre partition for hypergraphs and multigraphs built from the same data.
- 'Synchronization_simulations/': folder containing Jupyter Notebooks with examples of simulation of cluster synchronization dynamics on hypergraphs and multigraphs and their comparison with the fibres.
- 'Topology_modifications/': folder containing Jupyter Notebooks with examples of how to modify the topology of hypergraphs to cancel or add hyperedges preserving or forcing the fibre partition.


## Dependencies
To visualize and realize the test hypergraphs we use the Python library [Hypergraphx](https://hypergraphx.readthedocs.io/en/master/index.html).
To assign a color in the visualization we modified their script __draw_hypergraph_col.py__ to assign particular colors to the hypergraph nodes.

The datasets (apart from MAG-10) used are from [XGI library](https://github.com/ComplexGroupInteractions/xgi).
