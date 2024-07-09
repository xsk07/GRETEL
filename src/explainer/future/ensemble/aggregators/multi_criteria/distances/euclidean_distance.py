import numpy as np

from src.explainer.future.ensemble.aggregators.multi_criteria.distances.base_distance import (
    BaseDistance,
)


class EuclideanDistance(BaseDistance):
    def calculate(
        self,
        matrix: np.ndarray,
        vector: np.ndarray,
    ) -> np.ndarray:
        return np.linalg.norm(matrix - vector, axis=1)
