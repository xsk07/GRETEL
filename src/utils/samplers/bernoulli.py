from types import SimpleNamespace
from typing import List

import torch

from src.core.oracle_base import Oracle
from src.utils.samplers.abstract_sampler import Sampler
from src.dataset.instances.graph import GraphInstance
from src.utils.utils import pad_features


class Bernoulli(Sampler):
    
    def __init__(self, sampling_iterations=1):
        super().__init__()
        self._name = 'bernoulli'
        self.sampling_iterations = sampling_iterations
          
    def sample(self, instance: GraphInstance, oracle: Oracle, **kwargs) -> List[GraphInstance]:
        kwargs = SimpleNamespace(**kwargs)
        edge_probs = kwargs.edge_probabilities
        embedded_features = kwargs.embedded_features

        pred_label = oracle.predict(instance)
        edge_probs = edge_probs.get(pred_label)
        node_features = embedded_features.get(pred_label).cpu().numpy()
        
        for _ in range(self.sampling_iterations):
            cf_instance = self.__sample(instance, node_features, edge_probs)
            if oracle.predict(cf_instance) != pred_label:
                return cf_instance
            
        return None 
                
    def __sample(self, instance: GraphInstance, features, probabilities):
        adj = torch.bernoulli(probabilities)
        try:
            adj.shape[1]
        except IndexError:
            adj = adj.reshape(features.shape[0], features.shape[0])
    
        features = pad_features(features, adj.shape[0])            
        selected_edges = torch.nonzero(adj)
        try:
            probabilities.shape[1]
        except:
            probabilities = probabilities.reshape(features.shape[0], features.shape[0])

        cf_candidate = GraphInstance(id=instance.id,
                                     label=1-instance.label,
                                     data=adj.numpy(),
                                     node_features=features,
                                     edge_weights=probabilities[selected_edges[:,0], selected_edges[:,1]].numpy())
        
        instance._dataset.manipulate(cf_candidate)
     
        return cf_candidate