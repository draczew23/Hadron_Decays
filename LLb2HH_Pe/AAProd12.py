import numpy as np
from math import *
class AAProd12:
    def __init__(self, aP, pP, Pe):
        self.aP = aP
        self.pP = pP
        self.Pe = Pe
        self.v = sqrt(1-aP**2)
        self.bP = self.v*sin(pP)
        self.gP = self.v*cos(pP)
        self.C = np.zeros((4,4))
    
    def get_kine_calc_trig(self, th):
        """_This function takes kinematic parameters, saves those parameters
        as class instance attributes and calculates trigonometric functions.
        Those functions are also saved as class instance attributes._

        Args:
            th (_float_): _kinematic parameter_
        """

        self.st = sin(th)
        self.ct = cos(th)

    def create_matrix(self):
        """_This functions creates a decay matrix. Martix elements
        are filled via class instance atributes._

        Returns:
            _ndarray_: _a decay matrix_
        """

        aP = self.aP
        bP = self.bP
        gP = self.gP
        Pe = self.Pe
        st = self.st
        ct = self.ct
        ct2 = ct**2
        self.C[0, 0] = 1 + aP*ct2
        self.C[0, 1] = gP*Pe*st
        self.C[0, 2] = bP*st*ct
        self.C[0, 3] = (1 + aP)*Pe*ct
        self.C[1, 0] = gP*Pe*st
        self.C[1, 1] = 1 - ct2
        self.C[1, 3] = gP*st*ct
        self.C[2, 0] = -bP*st*ct
        self.C[2, 2] = aP*(1 - ct2)
        self.C[2, 3] = -bP*Pe*st
        self.C[3, 0] = -(1 + aP)*Pe*ct
        self.C[3, 1] = -gP*st*ct
        self.C[3, 2] = -bP*Pe*st
        self.C[3, 3] = -aP-ct2

        return self.C

