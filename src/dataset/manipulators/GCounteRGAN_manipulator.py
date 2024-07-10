import networkx as nx
import numpy as np

from src.dataset.manipulators.base import BaseManipulator

#manipulator for GCounterGAN explainer
class PaddingGCounteRGAN(BaseManipulator):
    
    
    def node_info(self, instance):
        adj = instance.data
        n_nodes=adj.shape[0]
        mult=4

        print("shape of node features")
        print(instance.node_features.shape)
        
        if (n_nodes==0):
            num_padding=mult
        else:
            num_padding=(mult-(n_nodes%mult))%mult#((n_nodes + 3) // 4) * 4 - n_nodes
            
        instance.data=np.pad(instance.data,((0,num_padding),(0,num_padding)),'constant',constant_values=0)
        instance.node_features=instance.node_features.reshape(1,instance.node_features.shape[0] ) 
        
        #instance.node_features = np.pad(instance.node_features, ((0, num_padding)), mode='constant', constant_values=0)
        instance.node_features=np.pad(instance.node_features,((0,num_padding),(0,0)),'constant',constant_values=0)
       # dim= int(instance.node_features.shape[1])
        #instance.node_features=instance.node_features.reshape(-1, )
        #print("new shape of node features")
        #print(instance.node_features.shape)
        return {}
        #print(adj.shape)
        
        return {}