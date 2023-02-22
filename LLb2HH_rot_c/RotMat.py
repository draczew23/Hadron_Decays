import numpy as np
from math import *
class RotMat:
    def __init__(self, theta=0, chi=0, phi=0):
        """ _Class to create a rotation matrix in 4 dimensions based
        on the Euler angles (in radians) provided by the user.
        
        Args:
            theta (_float_): _the signed angle between the z axis and the Z axis_
            chi (_float_): _the signed angle between the N axis and the X axis_
            phi (_float_): _the signed angle between the x axis and the N axis_
        """
        self.theta = theta
        self.chi = chi
        self.phi = phi
        self.R = np.zeros((4, 4))
    
    def create(self):
        """_This function creates a rotation matrix based on the Euler angles_
        """
        theta = self.theta
        chi = self.chi
        phi = self.phi
       
        cth = np.cos(theta)
        sth = np.sin(theta)
        cch = np.cos(chi)
        sch = np.sin(chi)
        cph = np.cos(phi)
        sph = np.sin(phi)
        
        self.R[0, 0] = 1
        self.R[1, 1] = cth*cch*cph - sch*sph
        self.R[1, 2] = -cth*sch*cph - cch*sph
        self.R[1, 3] = sth*cph
        self.R[2, 1] = cth*cch*sph + sch*cph
        self.R[2, 2] = cch*cph - cth*sch*sph
        self.R[2, 3] = sth*sph
        self.R[3, 1] = -sth*cph
        self.R[3, 2] = sth*sch
        self.R[3, 3] = cth

        return self.R
