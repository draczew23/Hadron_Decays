import numpy as np
from math import *
from RotMat import *
class AADecay12new:
    def __init__(self, aX, aY, aZ):
        self.aX = aX
        self.aY = aY
        self.aZ = aZ
        self.A = np.zeros((4, 4))
        self.A[0, 0] = 1
    
    def create_matrix(self):
        """_This functions creates a decay matrix. Martix elements
        are filled via class instance atributes._

        Returns:
            _ndarray_: _a decay matrix_
        """

        aX = self.aX
        aY = self.aY
        aZ = self.aZ

        self.A[1, 0] = aX
        self.A[2, 0] = aY
        self.A[3, 0] = aZ

        return self.A

