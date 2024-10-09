import networkx as nx
import numpy as np
import community as community_louvain
from scipy.linalg import eigh

class GraphMetrics:
    def __init__(self, adj_matrix):
        """
        Initialize the GraphMetrics class with an adjacency matrix.

        Parameters
        ----------
        adj_matrix : numpy.ndarray
            Weighted adjacency matrix of the graph.
        """
        # Ensure weights are non-negative by replacing negative values with zero
        adj_matrix = np.maximum(adj_matrix, 0)
        self.graph = nx.from_numpy_array(adj_matrix, create_using=nx.Graph)
        self.adj_matrix = adj_matrix  # Keep the original adjacency matrix for calculations

    def calculate_degree_std(self):
        """
        Calculate the standard deviation of the weighted degree of the graph.

        Returns
        -------
        float
            The standard deviation of the weighted degree of the nodes in the graph.
        """
        # Calculate the weighted degree of each node
        weighted_degrees = [degree for node, degree in self.graph.degree(weight='weight')]
        
        # Calculate the standard deviation of the weighted degrees
        degree_std = np.std(weighted_degrees) if weighted_degrees else 0
        
        return degree_std

    def calculate_degree_centrality(self):
        """
        Calculate the degree centrality for each node considering the weights.

        Returns
        -------
        dict
            Degree centrality for each node.
        """
        return nx.degree_centrality(self.graph)

    def calculate_betweenness_centrality(self):
        """
        Calculate the betweenness centrality for each node considering the weights.

        Returns
        -------
        dict
            Betweenness centrality for each node.
        """
        return nx.betweenness_centrality(self.graph, weight='weight')

    def calculate_closeness_centrality(self):
        """
        Calculate the closeness centrality for each node considering the weights.

        Returns
        -------
        dict
            Closeness centrality for each node.
        """
        return nx.closeness_centrality(self.graph, distance='weight')

    def calculate_eigenvector_centrality(self):
        """
        Calculate the eigenvector centrality for each node considering the weights.

        Returns
        -------
        dict
            Eigenvector centrality for each node.
        """
        return nx.eigenvector_centrality_numpy(self.graph, weight='weight')

    def calculate_local_efficiency(self):
        """
        Calculate the local efficiency for each node.

        Returns
        -------
        dict
            Local efficiency for each node.
        """
        return {node: nx.local_efficiency(self.graph.subgraph(self.graph[node])) for node in self.graph}

    def calculate_global_efficiency(self):
        """
        Calculate the global efficiency of the graph.

        Returns
        -------
        float
            Global efficiency of the graph.
        """
        return nx.global_efficiency(self.graph)

    def calculate_average_clustering(self):
        """
        Calculate the average clustering coefficient of the graph considering weights.

        Returns
        -------
        float
            Average clustering coefficient of the graph.
        """
        return nx.average_clustering(self.graph, weight='weight')

    def calculate_small_worldness(self):
        """
        Calculate the small-worldness of the graph.

        Returns
        -------
        float
            Small-worldness of the graph.
        """
        clustering_coefficient = nx.average_clustering(self.graph, weight='weight')
        path_length = nx.average_shortest_path_length(self.graph, weight='weight')
        random_graph = nx.gnm_random_graph(len(self.graph.nodes), len(self.graph.edges))
        random_clustering_coefficient = nx.average_clustering(random_graph)
        random_path_length = nx.average_shortest_path_length(random_graph)
        return (clustering_coefficient / random_clustering_coefficient) / (path_length / random_path_length)

    def calculate_modularity(self):
        """
        Calculate the modularity of the graph.

        Returns
        -------
        float
            Modularity of the graph.
        """
        partition = community_louvain.best_partition(self.graph)
        return community_louvain.modularity(partition, self.graph)

    def calculate_assortativity(self):
        """
        Calculate the assortativity of the graph considering weights.

        Returns
        -------
        float
            Assortativity of the graph.
        """
        return nx.degree_assortativity_coefficient(self.graph, weight='weight')

    def calculate_all_node_metrics(self):
        """
        Calculate all node-level metrics.

        Returns
        -------
        dict
            Dictionary containing all node-level metrics.
        """
        return {
            'degree_centrality': self.calculate_degree_centrality(),
            'betweenness_centrality': self.calculate_betweenness_centrality(),
            'closeness_centrality': self.calculate_closeness_centrality(),
            'eigenvector_centrality': self.calculate_eigenvector_centrality(),
            'local_efficiency': self.calculate_local_efficiency(),
        }

    def calculate_all_graph_metrics(self):
        """
        Calculate all whole-graph metrics.

        Returns
        -------
        dict
            Dictionary containing all whole-graph metrics.
        """
        return {
            'global_efficiency': self.calculate_global_efficiency(),
            'average_clustering': self.calculate_average_clustering(),
            'small_worldness': self.calculate_small_worldness(),
            'modularity': self.calculate_modularity(),
            'assortativity': self.calculate_assortativity(),
        }
    

    def calculate_segregation(self, nodes_in_community):
        """
        Calculate the segregation measure for a subnetwork (community).

        Parameters
        ----------
        nodes_in_community : array-like
            Indices of nodes belonging to the community.

        Returns
        -------
        float
            The segregation measure for the community.
        """
        # Mean intra-network connectivity (within the subnetwork)
        intra_network_connectivity = self.adj_matrix[np.ix_(nodes_in_community, nodes_in_community)]
        mean_intra_network = np.mean(intra_network_connectivity)

        # Mean inter-network connectivity (between the subnetwork and all other nodes)
        inter_network_connectivity = self.adj_matrix[np.ix_(nodes_in_community, np.setdiff1d(np.arange(self.adj_matrix.shape[0]), nodes_in_community))]
        mean_inter_network = np.mean(inter_network_connectivity)

        # Calculate segregation as the ratio of intra-network to inter-network connectivity
        if mean_inter_network == 0:  # Avoid division by zero
            return np.inf if mean_intra_network > 0 else 0
        else:
            return mean_intra_network / mean_inter_network
        
    def calculate_spectral_entropy(self, subgraph):
        """
        Calculate the spectral entropy of a given subgraph.

        Parameters
        ----------
        subgraph : networkx.Graph
            The subgraph for which to calculate the spectral entropy.

        Returns
        -------
        float
            The spectral entropy of the subgraph.
        """
        # Step 1: Compute the Laplacian matrix L = D - W
        adj_matrix_sub = nx.to_numpy_array(subgraph)
        degree_matrix_sub = np.diag(np.sum(adj_matrix_sub, axis=1))
        laplacian_matrix_sub = degree_matrix_sub - adj_matrix_sub

        # Step 2: Compute the eigenvalues of the Laplacian matrix
        eigenvalues = eigh(laplacian_matrix_sub, eigvals_only=True)

        # Step 3: Normalize the eigenvalues to create a probability distribution
        eigenvalues = np.maximum(eigenvalues, 1e-12)  # Avoid log(0) by setting a small value
        total_sum = np.sum(eigenvalues)
        probabilities = eigenvalues / total_sum

        # Step 4: Compute the spectral entropy
        spectral_entropy = -np.sum(probabilities * np.log(probabilities))

        return spectral_entropy
    
    def calculate_average_path_length_subgraph(self, subgraph):
        """
        Calculate the average path length of a given subgraph.

        Parameters
        ----------
        subgraph : networkx.Graph
            The subgraph for which to calculate the average path length.

        Returns
        -------
        float
            The average path length of the subgraph.
        """
        if nx.is_connected(subgraph):
            return nx.average_shortest_path_length(subgraph, weight='weight')
        else:
            # For disconnected subgraphs, calculate the average path length of each component and return the weighted average
            components = nx.connected_components(subgraph)
            total_sum = 0
            total_count = 0
            for component in components:
                component_subgraph = subgraph.subgraph(component)
                component_size = len(component)
                if component_size > 1:  # Avoid calculating path length for a single-node component
                    total_sum += nx.average_shortest_path_length(component_subgraph, weight='weight') * component_size
                    total_count += component_size
            return total_sum / total_count if total_count > 0 else float('inf')
        

    def calculate_community_metrics(self, df):
        """
        Calculate graph metrics for each community defined in the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing 'ParcelID' and 'Community' columns.

        Returns
        -------
        dict
            A dictionary where keys are community names and values are dictionaries
            containing graph metrics for the corresponding community.
        """
        community_metrics = {}
        communities = df['Community'].unique()

        for community in communities:
            nodes_in_community = df[df['Community'] == community]['ParcelID'].values - 1
            subgraph = self.graph.subgraph(nodes_in_community)

            metrics = {
                    'segregation': self.calculate_segregation(nodes_in_community),
                    'average_clustering': nx.average_clustering(subgraph, weight='weight') if subgraph.number_of_nodes() > 1 else 0,
                    'average_betweenness_centrality': np.mean(list(nx.betweenness_centrality(subgraph, weight='weight').values())) if subgraph.number_of_nodes() > 0 else 0,
                    #'spectral_entropy': self.calculate_spectral_entropy(subgraph),
                    'average_path_lenght' : self.calculate_average_path_length_subgraph(subgraph),
                    'average_degree' : self.calculate_average_degree(subgraph)
                    }

            community_metrics[community] = metrics

        return community_metrics

    def calculate_average_degree(self, subgraph):
        """
        Calculate the average weighted degree of the network.

        Parameters
        ----------
        subgraph : networkx.Graph
            The subgraph for which to calculate the average weighted degree.

        Returns
        -------
        float
            The average weighted degree of the nodes in the subgraph.
        """
        # Calculating the weighted degree of each node
        weighted_degrees = [degree for node, degree in subgraph.degree(weight='weight')]
        
        # Calculating the average of the weighted degrees
        average_weighted_degree = np.mean(weighted_degrees) if weighted_degrees else 0
        
        return average_weighted_degree

    def calculate_global_metrics(self):
        """
        Calculate global graph metrics.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            A dictionary where keys are metric names and values are graph metrics for the corresponding graph.
        """

        subgraph = self.graph

        metrics = {
            'average_clustering': nx.average_clustering(subgraph, weight='weight') if subgraph.number_of_nodes() > 1 else 0,
            'average_betweenness_centrality': np.mean(list(nx.betweenness_centrality(subgraph, weight='weight').values())) if subgraph.number_of_nodes() > 0 else 0,
            #'spectral_entropy': self.calculate_spectral_entropy(subgraph),
            'average_path_length': self.calculate_average_path_length_subgraph(subgraph),
            'average_degree': self.calculate_average_degree(subgraph),  # New metric
            'degree_std': self.calculate_degree_std()  # New metric for standard deviation of degree

        }

        return metrics
