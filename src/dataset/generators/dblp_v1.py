import numpy as np
import pandas as pd
from io import StringIO

from os.path import join
from src.dataset.generators.base import Generator
from src.dataset.instances.graph import GraphInstance

class DBLP(Generator):

    def init(self):
        #configuration parameters
        base_path = self.local_config['parameters']['data_dir']
        self.dataset_name = self.local_config['parameters']['dataset_name']

        #Paths to the files of the DBLP_v1 dataset
        self._adj_file_path = join(base_path, f"{self.dataset_name}_A.txt")
        self._gid_file_path = join(base_path, f"{self.dataset_name}_graph_indicator.txt")
        self._gl_file_path = join(base_path, f"{self.dataset_name}_graph_labels.txt")
        self._nl_file_path = join(base_path, f"{self.dataset_name}_node_labels.txt")
        self._el_file_path = join(base_path, f"{self.dataset_name}_edge_labels.txt")
        self.readme_path = join(base_path, f"readme.txt")

        # Node type is 'paper' or 'keyword'
        self.dataset.node_features_map = {'node_type': 0, 'value': 1}
        # Paper to Paper, Paper to Word, Word to Word
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
            edges = np.genfromtxt(self._adj_file_path, dtype=int, delimiter=',')
            edge_labels = np.loadtxt(self._el_file_path, dtype=int)
            graph_ind = np.loadtxt(self._gid_file_path, dtype=int)
            node_features = self.get_node_features(self.readme_path, self._nl_file_path)

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

                #if the matrix is singular then ignore the instance
                if not np.isclose(np.linalg.det(adj_matrix), 0):
                    self.dataset.instances.append(
                        GraphInstance(
                            id,
                            label=labels[id-1],
                            data=adj_matrix,
                            dataset=self.dataset,
                            node_features=node_features[node_mask],
                            edge_features=np.eye(3)[edge_labels[np.any(edge_mask, axis=1)]] # one-hot encoding of the edge lables
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

    def read_mapping(self, file_path):
        # string indentifying the beginning of the mapping
        start_sentence = "Component 0:"

        data_lines = [] # to store the lines after the target sentence
        # read the file line by line
        with open(file_path, 'r') as file:
            start_reading = False
            for line in file:
                # check if the line starts with the sentence
                if line.startswith(start_sentence):
                    start_reading = True

                # once the target sentence is found, collect lines until a blank line is found
                if start_reading:
                    # stop when a blank line is encountered
                    if line.strip() == "":
                        break
                    # append non-blank lines
                    data_lines.append(line.strip())

        # convert list of lines into a single string
        data_str = '\n'.join(data_lines)

        # read data into a DataFrame
        return pd.read_csv(StringIO(data_str), sep='\s+') 

    def normalize(self, values):
        #min-max normalization
        min_val = np.min(values)
        max_val = np.max(values)
        return (values-min_val)/(max_val-min_val)

    def get_node_features(self, readme_path, nl_file_path):

        mapping = self.read_mapping(readme_path)
        # 'value' stands for target value, 'original' stands for original value before the mapping
        mapping.rename(columns={mapping.columns[0]: 'value', mapping.columns[1]: 'original'}, inplace=True)

        node_labels = pd.read_csv(nl_file_path, header=None)
        node_labels.rename(columns={node_labels.columns[0] : 'value'}, inplace=True)

        node_features = pd.merge(node_labels, mapping, on='value', how='left')

        # node type: paper or keyword
        node_features['type'] = pd.to_numeric(node_features.original, errors='coerce')
        node_features['type'] = 1-node_features.type.isna().astype(int) # 1 if paper id, 0 if keyword
        
        node_features['value'] = self.normalize(node_features['value'])
        assert np.all(node_features['value'].between(0,1)) #values are in [0,1]

        return node_features[['type', 'value']].values