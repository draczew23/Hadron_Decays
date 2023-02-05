import numpy as np
from math import *
from RotMat import *
from AADecay12new import *

class derivative_calculator_decay:
        def __init__(self):
            pass

        def calc_derivatives(self, vars, ind, h):
            m = AADecay12new(vars[0], vars[1], vars[2])
            vars_step = vars
            vars_step[ind] += h
            m_shifted = AADecay12new(vars_step[0], vars_step[1], vars_step[2])

            m_matrix = m.create_matrix()
            m_shifted_matrix = m_shifted.create_matrix()

            derimat = (m_shifted_matrix - m_matrix) / h
            
            del m
            del m_shifted
            del m_matrix
            del m_shifted_matrix

            return derimat

