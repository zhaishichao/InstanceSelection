import numpy as np

from instance_selection_encapsulation.MILE.config import DatasetConfig


# 类型注解 dataset: DatasetConfig
class MILE():
    def __init__(self, dataset=None, estimator=None, random_state=42):
        self.dataset = dataset,
        self.estimator = estimator,
        self.random_state = random_state

