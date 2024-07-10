import networkx as nx
import numpy as np

from src.dataset.manipulators.base import BaseManipulator


class CLEARPadding(BaseManipulator):
    
    def node_info(self, instance):
        
        padding= max(self.dataset.num_nodes_values)-instance.data.shape[0] 
        
        instance.data=np.pad(instance.data,((0,padding),(0,padding)),'constant',constant_values=0)
        instance.node_features=np.pad(instance.node_features,((0,padding),(0,0)),'constant',constant_values=0)
        return {}
