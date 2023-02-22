import numpy as np
from math import *
class AAProd12_alt:
    def __init__(self, Pz):
        self.Pz = Pz
        self.C = np.array([1, 0, 0, Pz])
        self.dCPz = np.array([0, 0, 0, 1])

