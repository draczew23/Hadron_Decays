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
    
    def get_kine_calc_trig(self, th, ph, ch):
        """_This function takes kinematic parameters, saves those parameters
        as class instance attributes and calculates trigonometric functions.
        Those functions are also saved as class instance attributes._

        Args:
            th (_float_): _kinematic parameter_
            ph (_float_): _kinematic parameter_
            ch (_float_): _kinematic parameter_
        """

        self.th = th
        self.ph = ph
        self.ch = ch
        self.RM = np.zeros((4,4))

    
    def create_matrix(self):
        """_This functions creates a decay matrix. Martix elements
        are filled via class instance atributes._

        Returns:
            _ndarray_: _a decay matrix_
        """

        aX = self.aX
        aY = self.aY
        aZ = self.aZ

        self.vec4 = np.array([1, aX, aY, aZ])

        th = self.th
        ph = self.ph
        ch = self.ch
        RM = RotMat(th, ph, ch)
        R = RM.create()

        self.A = np.matmul(R, self.vec4)

        # self.A[1, 1] = R[1,:]*aX
        # self.A[2, 2] = R[2,:]*aY
        # self.A[3, 3] = R[3,:]*aZ

        return self.A

