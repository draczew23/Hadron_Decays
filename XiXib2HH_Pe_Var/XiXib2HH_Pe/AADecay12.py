import numpy as np
from math import *
class AADecay12:
    def __init__(self, aM, pM=0):
        self.aM = aM
        self.v = sqrt(1-aM**2)
        self.bM = self.v*sin(pM)
        self.gM = self.v*cos(pM)
        self.A = np.zeros((4,4))
    
    def get_kine_calc_trig(self, th, ph):
        """_This function takes kinematic parameters, saves those parameters
        as class instance attributes and calculates trigonometric functions.
        Those functions are also saved as class instance attributes._

        Args:
            th (_float_): _kinematic parameter_
            ph (_float_): _kinematic parameter_
        """

        self.sp = sin(ph)
        self.cp = cos(ph)
        self.st = sin(th)
        self.ct = cos(th)

    def create_matrix(self):
        """_This functions creates a decay matrix. Martix elements
        are filled via class instance atributes._

        Returns:
            _ndarray_: _a decay matrix_
        """

        aM = self.aM
        bM = self.bM
        gM = self.gM
        sp = self.sp
        cp = self.cp
        st = self.st
        ct = self.ct
        self.A[0, 0] = 1
        self.A[0, 3] = aM
        self.A[1, 0] = aM*cp*st
        self.A[1, 1] = gM*ct*cp-bM*sp
        self.A[1, 2] = -bM*ct*cp-gM*sp
        self.A[1, 3] = cp*st
        self.A[2, 0] = aM*sp*st
        self.A[2, 1] = bM*cp+gM*ct*sp
        self.A[2, 2] = gM*cp-bM*ct*sp
        self.A[2, 3] = sp*st
        self.A[3, 0] = aM*ct
        self.A[3, 1] = -gM*st
        self.A[3, 2] = bM*st
        self.A[3, 3] = ct

        return self.A

