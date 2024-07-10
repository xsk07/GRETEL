import networkx as nx
import numpy as np
from src.dataset.manipulators.base import BaseManipulator

class PaddingGCounteRGAN(BaseManipulator):
    
    
    def node_info(self, instance):
        adj = instance.data
        n_nodes=adj.shape[0]
        mult=4
        
        if (n_nodes==0):
            num_padding=mult
        else:
            num_padding=(mult-(n_nodes%mult))%mult
            
        instance.data=np.pad(instance.data,((0,num_padding),(0,num_padding)),'constant',constant_values=0)
        instance.node_features=np.pad(instance.node_features,((0,num_padding),(0,0)),'constant',constant_values=0)
       
        return {}
        