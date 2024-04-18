import numpy as np
from gluonts.transform import MapTransformation 

class Transpose(MapTransformation):

    def __init__(self, field: str, target_dim = None) -> None:
        self.field = field
        self.target_dim = target_dim

    def map_transform(self, data: np.ndarray, is_train: bool):
        data[self.field] = np.transpose(data[self.field], self.target_dim)
        return data