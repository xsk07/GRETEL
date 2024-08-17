from src.utils.context import Context
from src.core.configurable import Configurable
from src.core.grtl_base import Base
from src.dataset.dataset_base import Dataset
from src.core.oracle_base import Oracle
from src.core.explainer_base import Explainer


class Explanation(Base):
    """The common logic shared between all explanation types should be in this class"""
    
    def __init__(self, context: Context, dataset: Dataset, oracle: Oracle, explainer: Explainer) -> None:
        self._context = context
        self._dataset = dataset
        self._oracle = oracle
        self._explainer = explainer
        self._stages_info = {}
        self._info = {}

        #TODO Deprecate this dictionaries
        # self._metrics_info = {}
        # self._runtime_info = {} 

    @property
    def context(self) -> Context:
        return self._context

    @context.setter
    def context(self, new_context) -> None:
        self._context = new_context

    
    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @dataset.setter
    def dataset(self, new_dataset) -> None:
        self._dataset = new_dataset

    
    @property
    def oracle(self) -> Oracle:
        return self._oracle

    @oracle.setter
    def oracle(self, new_oracle) -> None:
        self._oracle = new_oracle


    @property
    def explainer(self) -> Explainer:
        return self._explainer

    @explainer.setter
    def explainer(self, new_explainer) -> None:
        self._explainer = new_explainer
        
    @property
    def stages_info(self) -> dict:
        return self._stages_info
    
    @stages_info.setter
    def stages_info(self, new_stages_info) -> None:
        self._stages_info = new_stages_info
    
    @property
    def info(self) -> dict:
        return self._info
    
    @info.setter
    def info(self, new_info) -> None:
        self._info = new_info
    
    # TODO Deprecate this properties
    # @property
    # def runtime_info(self) -> dict:
    #     return self._runtime_info
    
    # @runtime_info.setter
    # def runtime_info(self, new_runtime_info) -> None:
    #     self._runtime_info = new_runtime_info

    
    @property
    def explanation_runtime(self) -> int:
        if 'explanation_runtime' not in self._runtime_info:
            return -1
        return self._runtime_info['explanation_runtime']
    
    @explanation_runtime.setter
    def explanation_runtime(self, new_runtime_info) -> None:
        self._runtime_info['explanation_runtime'] = new_runtime_info

    