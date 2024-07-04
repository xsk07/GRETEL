from os.path import join
import numpy as np

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance


class DBLP(Generator):

    def init(self):
        #configuration parameters
        base_path = self.local_config['parameters']['data_dir']
        self.dataset_name = self.local_config['parameters']['dataset_name']
        #self.min_nodes = self.local_config['parameters']['min_nodes']
        #self.max_nodes = self.local_config['parameters']['max_nodes']

        # Paths to the files of the DBLP_v1 dataset
        self._adj_file_path = join(base_path, f"{self.dataset_name}_A.txt")
        self._gid_file_path = join(base_path, f"{self.dataset_name}_graph_indicator.txt")
        self._gl_file_path = join(base_path, f"{self.dataset_name}_graph_labels.txt")
        self._nl_file_path = join(base_path, f"{self.dataset_name}_node_labels.txt")
        self._el_file_path = join(base_path, f"{self.dataset_name}_edge_labels.txt")

        #self.dataset.node_features_map = {} # ???
        #self.dataset.edge_features_map = {} # ???
        self.generate_dataset()
            
    def generate_dataset(self):
        
        if not self.get_num_instances():
            
            #load the dataset
            labels = np.loadtxt(self._gl_file_path, dtype=int)
            node_labels = np.loadtxt(self._nl_file_path, dtype=int)
            edges = np.genfromtxt(self._adj_file_path, dtype=int, delimiter=',')
            edge_labels = np.loadtxt(self._el_file_path, dtype=int)
            graph_ind = np.loadtxt(self._gid_file_path, dtype=int)

            assert np.all(np.diff(graph_ind)>=0) #the array of graph indicators is monotone increasing
            #it means that we can subtract the min node of each graph to its nodes,
            #in this way we obtain smaller adjacency matrices for all the graphs

            #renaming the nodes so that are numbered from 0 to n-1
            edges = edges-1
            #list of edges in which each node's number is replaced with the graph id it belongs to
            edges_gid = graph_ind[edges]

            graph_ids = np.unique(graph_ind) #list of graph identifiers
            for id in graph_ids:

                #masks for filtering accordingly to the graph identifier
                node_mask = (graph_ind == id) #mask for nodes
                edge_mask = (edges_gid == id) #mask for edges

                filtered_edges = edges[np.any(edge_mask, axis=1)] #select edges of the graph with identifier id
                assert np.all(np.diff(np.unique(filtered_edges))==1)
                min_node = filtered_edges.min() #number identifier of the minimum node in the subgraph
                #subtract the minimum node to the edges, so that nodes are numbered from 0 to k,
                #k is the maximum node of the mapping
                #This step allows us to reduce the size of the adjacency matrix
                #the adjacency matrix now is quadratic in the size of the subgraph
                mapped_edges = filtered_edges-min_node
                assert np.all(np.diff(np.unique(mapped_edges))==1)

                adj_matrix = self.create_adj_matrix(mapped_edges)
                # manages singluar matrixes,
                # we add a small constant to the diagonal so that the determinant is non-zero anymore
                # if np.isclose(np.linalg.det(adj_matrix), 0):
                #adj_matrix += 1e-10 * np.eye(adj_matrix.shape[0])
                # ensures that the matrix is non-singular
                # assert not np.isclose(np.linalg.det(adj_matrix), 0), f"singular matrix found!"

                if not np.isclose(np.linalg.det(adj_matrix), 0):
                    self.dataset.instances.append(
                        GraphInstance(
                            id,
                            label=labels[id-1],
                            data=self.create_adj_matrix(mapped_edges),
                            node_features=node_labels[node_mask],
                            edge_features=edge_labels[np.any(edge_mask, axis=1)]
                        )
                    )


    def create_adj_matrix(self, edges):
        # after the mapping, nodes are numbered from 0 to max
        num_nodes = np.max(edges)+1 
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        # The edge list takes into account that the edges are undirected,
        # so already it contains the edges of the two possible permutations of nodes,
        # and so there is no need to insert a 1 in the transposed positions
        adj_matrix[edges[:,0], edges[:,1]] = 1
        return adj_matrix    