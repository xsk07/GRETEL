from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src.utils.cfg_utils import default_cfg
from src.utils.torch.graph_pooling import TopKPooling

class GraphEmbedder(nn.Module):
    
    def __init__(self, num_nodes, node_feature_dim, dim=2):
        """This class provides a GCN to discriminate between real and generated graph instances"""
        super(GraphEmbedder, self).__init__()

        self.training = False
        
        self.conv = GCNConv(node_feature_dim, dim).double()
        self.pool = TopKPooling(in_channels=dim, k=num_nodes)
        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        
    def set_training(self, training):
        self.training = training

    def forward(self, x, edge_list, edge_attr, batch=None):
        x = x.double()
        edge_attr = edge_attr.double()
        x = self.conv(x, edge_list, edge_attr)
        x = F.relu(x)
        x, _, _, _, _, _ = self.pool(x, edge_list, edge_attr, batch)
        return x
    
    @default_cfg
    def grtl_default(kls, num_nodes, node_feature_dim, dim=2):
        return {"class": kls,
                        "parameters": {
                            "num_nodes": num_nodes,
                            "node_feature_dim": node_feature_dim,
                            "dim": dim
                        }
        }
        

class EdgeExistanceModule(nn.Module):

    def __init__(self, dim=2) -> None:
        super(EdgeExistanceModule, self).__init__()
        # decodes the edge embeddings (concatenation of two node vectors)
        self.edge_decoder = nn.Linear(2 * dim, 1)

    def forward(self, emb_nodes, edge_list) -> Tuple[torch.Tensor,
                                                     torch.Tensor,
                                                     torch.Tensor]:
        def tensor_in_list(tensor, tensor_list):
            return any(torch.equal(tensor, t) for t in tensor_list)
        
        # repeat the node embeddings n times where n is the number of nodes 
        interleaved = torch.repeat_interleave(emb_nodes, repeats=emb_nodes.shape[0], dim=0).detach()
        repeated = emb_nodes.repeat(emb_nodes.shape[0], 1).detach()
        # where all rows are zero, then there's a self-loop, which we need to delete
        loops = interleaved - repeated
        loops = loops.detach()
        non_empty_mask = loops.abs().sum(dim=1).bool()
        # Initialize the real edges tensor
        real_edges = []
        for node1, node2 in list(zip(edge_list[0], edge_list[1])):
            real_edges.append(torch.concat((emb_nodes[node1], emb_nodes[node2])))
        # create edge embeddings
        edge_embeddings = torch.concat([interleaved, repeated], dim=1).detach()
        # check if the edge exists
        edge_logits = self.edge_decoder(edge_embeddings)
        logits_without_self_loops = edge_logits.clone().squeeze()
        logits_without_self_loops[~non_empty_mask] = -10

        true_edges = torch.empty(size=(edge_embeddings.shape[0], 1))
    
        for i, edge_embedding in enumerate(edge_embeddings):
            true_edges[i] = 1 if tensor_in_list(edge_embedding, real_edges) else 0
        
        return edge_embeddings, logits_without_self_loops, true_edges.squeeze()
    

class NodeDecoderModule(nn.Module):

    def __init__(self, num_nodes, node_feature_dim, dim=2) -> None:
        super(NodeDecoderModule, self).__init__()
        # embeds the graph into node vectors
        self.embedder = GraphEmbedder(num_nodes, node_feature_dim, dim)
        # decodes the embedded node vectors into the original node feature space
        self.node_decoder = nn.Linear(dim, node_feature_dim)

    def forward(self, emb_nodes) -> torch.Tensor:
        return self.node_decoder(emb_nodes)
   