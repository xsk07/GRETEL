from os.path import join
import numpy as np

from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance


class DBLP(Generator):

    def init(self):
        base_path = self.local_config['parameters']['data_dir']
        self.dataset_name = self.local_config['parameters']['dataset_name']
        #self.max_nodes = self.local_config['parameters']['max_nodes']

        # Paths to the files of the "DBLP_v1 dataset"
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
            
            labels = np.loadtxt(self._gl_file_path, dtype=int)
            node_labels = np.loadtxt(self._nl_file_path, dtype=int)
            edges = np.genfromtxt(self._adj_file_path, dtype=int, delimiter=',')
            edge_labels = np.loadtxt(self._el_file_path, dtype=int)
            graph_ind = np.loadtxt(self._gid_file_path, dtype=int)

            edges-=1 
            edges_gid = graph_ind[edges]

            graph_ids = np.unique(graph_ind)
            for id in graph_ids:
                node_mask = (graph_ind == id)
                edge_mask = (edges_gid == id)

                filtered_edges = edges[np.any(edge_mask, axis=1)]
                
                self.dataset.instances.append(
                    GraphInstance(
                        id,
                        label=labels[id-1],
                        data=self.create_adj_matrix(self.map_nodes(filtered_edges)),
                        node_features=node_labels[node_mask],
                        edge_features=edge_labels[np.any(edge_mask, axis=1)]
                    )
                )
    
    def map_nodes(self, edges):
        flat = edges.flatten()
        _, inverse_indices = np.unique(flat, return_inverse=True)
        mapped_edges = inverse_indices.reshape(edges.shape)
        return mapped_edges

    def create_adj_matrix(self, edges):
        num_nodes = np.max(edges)+1 
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        adj_matrix[edges[:,0], edges[:,1]] = 1
        adj_matrix[edges[:,0], edges[:,1]] = 1
        return adj_matrix