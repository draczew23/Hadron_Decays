from math import *
import numpy as np
import time
from numpy.linalg import multi_dot
from scipy.misc import derivative
from AADecay12 import *
from AAProd12 import *
from derivative_calculator_prod import *
from derivative_calculator_decay import *

class PDF:
# Default values of the global parameters:
    aPsi = 0.461
    pPsi = 0.74
    Pe = 0.
    aL  = 0.758
    aLb = -0.758
    exec_time_loop_pdf = 0
    whole_gg = 0
    mat_init = 0


    def __init__(self, PaPsi=aPsi, PpPsi=pPsi, PPe=Pe):
        self.aPsi = PaPsi
        self.pPsi = PpPsi
        self.Pe = PPe
        # self.DL0 = AADecay12(self.aL)
        # self.DLb0 = AADecay12(self.aLb)
        # self.CC = AAProd12(PaPsi,PpPsi,PPe)
    """
    An example class of probability distributions and interface to vegas
#
    Attributes:
    aPsi (_float_): _alpha Jpsi decay parameter_
    pPsi (_float_): _phi Jpsi relative phase_
    Pe (_float_): _polarization electron beam_
    aL (_float_): _alpha Lambda decay parameter_
    aLb (_float_): _alpha anti-Lambda decay parameter_
    """

    @property
    def DL0(self):
        return AADecay12(self.aL)
    
    @property
    def DLb0(self):
        return AADecay12(self.aLb)

    @property
    def CC(self):
        return AAProd12(self.aPsi, self.pPsi, self.Pe)
    
k = PDF()
print(vars(k))
print(k.DL0.create_matrix())
