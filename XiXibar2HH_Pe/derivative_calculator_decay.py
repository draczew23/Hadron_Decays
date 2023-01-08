import numpy as np
from math import *
from AADecay12 import *

class derivative_calculator_decay:
        def __init__(self):
            pass
        def get_kinematic(self, th, ph):
            self.th = th
            self.ph = ph

        def calc_derivatives(self, vars, ind, h):
            m = AADecay12(vars[0], vars[1])
            vars_step = vars
            vars_step[ind] += h
            m_shifted = AADecay12(vars_step[0], vars_step[1])

            m.get_kine_calc_trig(self.th, self.ph)
            m_shifted.get_kine_calc_trig(self.th, self.ph)

            m_matrix = m.create_matrix()
            m_shifted_matrix = m_shifted.create_matrix()

            derimat = (m_shifted_matrix - m_matrix) / h
            
            del m
            del m_shifted
            del m_matrix
            del m_shifted_matrix

            return derimat

