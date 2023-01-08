import numpy as np
from math import *
from AAProd12 import *

class derivative_calculator_prod:
        def __init__(self):
            pass
        def get_kinematic(self, th):
            self.th = th

        def calc_derivatives(self, vars, ind, h):
            m = AAProd12(vars[0], vars[1], vars[2])
            vars_step = vars
            vars_step[ind] += h
            m_shifted = AAProd12(vars_step[0], vars_step[1], vars_step[2])

            m.get_kine_calc_trig(self.th)
            m_shifted.get_kine_calc_trig(self.th)

            m_matrix = m.create_matrix()
            m_shifted_matrix = m_shifted.create_matrix()

            derimat = (m_shifted_matrix - m_matrix) / h
            
            del m
            del m_shifted
            del m_matrix
            del m_shifted_matrix

            return derimat

