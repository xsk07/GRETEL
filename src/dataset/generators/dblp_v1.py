import numpy as np

from os.path import join
from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance

class DBLP(Generator):

    def init(self):
        #configuration parameters
        base_path = self.local_config['parameters']['data_dir']
        self.dataset_name = self.local_config['parameters']['dataset_name']

        #Paths to the files of the DBLP_v1 dataset
        #sparse (block diagonal) adjacency matrix for all graphs (node_id, node_id)
        self._adj_file_path = join(base_path, f"{self.dataset_name}_A.txt")
        #column vector of graph identifiers for all nodes of all graphs
        self._gid_file_path = join(base_path, f"{self.dataset_name}_graph_indicator.txt")
        #class labels for all graphs in the dataset
        self._gl_file_path = join(base_path, f"{self.dataset_name}_graph_labels.txt")
        #column vector of node labels
        self._nl_file_path = join(base_path, f"{self.dataset_name}_node_labels.txt")
        #labels for the edges in DS_A_sparse.txt
        self._el_file_path = join(base_path, f"{self.dataset_name}_edge_labels.txt")

        self.dataset.node_features_map = {'node_feature': 0}
        self.dataset.edge_features_map = {'P2P':0, 'P2W': 1, 'W2W': 2}
        self.generate_dataset()

    def check_configuration(self):
        super().check_configuration()
        local_config = self.local_config
        if 'dataset_name' not in local_config['parameters']:
            raise Exception("The name of the dataset must be given.")
            
    def generate_dataset(self):

        if not self.get_num_instances():

            #Loading the dataset
            labels = np.loadtxt(self._gl_file_path, dtype=int)
            #labels = labels.reshape((-1, 1))
            print("labels: ", labels.shape)
            node_labels = np.loadtxt(self._nl_file_path, dtype=int)
            node_labels = node_labels.reshape((-1, 1))
            print("node_labels: ", node_labels.shape)
            edges = np.genfromtxt(self._adj_file_path, dtype=int, delimiter=',')
            print("edges: ", edges.shape)
            edge_labels = np.loadtxt(self._el_file_path, dtype=int)
            #edge_labels = edge_labels.reshape((-1, 1))
            print("edge_labels: ", edge_labels.shape)
            graph_ind = np.loadtxt(self._gid_file_path, dtype=int)
            #graph_ind = graph_ind.reshape((-1, 1))
            print("graph_ind: ", graph_ind.shape)

            assert np.all(np.diff(graph_ind)>=0) #the array of graph indicators is monotone increasing
            #it means that we can subtract the min node of each graph to its nodes,
            #in this way we obtain smaller adjacency matrices for all the graphs

            #renaming the nodes so that are numbered from 0 to n-1
            edges = edges-1
            #list of edges in which each node's number is replaced with the graph id it belongs to
            edges_gid = graph_ind[edges]

            #min-max normalization
            min_val = np.min(node_labels)
            max_val = np.max(node_labels)
            node_labels = (node_labels-min_val)/(max_val-min_val)

            assert np.all((node_labels >= 0) & (node_labels <= 1)) #values of node labels are in [0,1]

            graph_ids = np.unique(graph_ind) #list of graph identifiers
            #np.random.shuffle(graph_ids)#shuffle randomly the graph ids
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

                #if the matrix is singular then ignore the instance
                if not np.isclose(np.linalg.det(adj_matrix), 0):

                    node_features = node_labels[node_mask]
                    # one-hot encoding of the edge lables
                    edge_features = np.eye(3)[edge_labels[np.any(edge_mask, axis=1)]]
                    
                    self.dataset.instances.append(
                        GraphInstance(
                            id,
                            label=labels[id-1],
                            data=adj_matrix,
                            dataset=self.dataset,
                            node_features=node_features,
                            edge_features=edge_features
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
        #adj_matrix[edges[:,1], edges[:,0]] = 1
        return adj_matrix