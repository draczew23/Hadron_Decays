from math import *
import numpy as np
import time
from numpy.linalg import multi_dot
from scipy.misc import derivative
from AADecay12new import *
# from AAProd12 import *

from AAProd12_alt import *


from RotMat import *
from derivative_calculator_prod import *
from derivative_calculator_decay import *
from RotationMatrix import *


class PDF:
# Default values of the global parameters:
    aPsi = 0.461
    pPsi = 0.74
    Pe = 0.
    aXL = 0.
    aYL = 0.
    aZL = 0.758
    aXLb = 0.
    aYLb = 0.
    aZLb = -0.758
    exec_time_loop_pdf = 0
    whole_gg = 0
    mat_init = 0


    def __init__(self, PaPsi=aPsi, PpPsi=pPsi, PPe=Pe):
        self.aPsi = PaPsi
        self.pPsi = PpPsi
        self.Pe = PPe
        self.DL0 = AADecay12new(self.aXL,self.aYL,self.aZL)
        self.DLb0 = AADecay12new(self.aXLb,self.aYLb,self.aZLb)
        self.CC = AAProd12_alt(0.3)
    """
    An example class of probability distributions and interface to vegas
#
    Attributes:
    aPsi (_float_): _alpha Jpsi decay parameter_
    pPsi (_float_): _phi Jpsi relative phase_
    Pe (_float_): _polarization electron beam_
    aXL (_float_): _aX Lambda decay parameter_
    aYL (_float_): _aY Lambda decay parameter_
    aZL (_float_): _aZ Lambda decay parameter_
    aXLb (_float_): _aX anti-Lambda decay parameter_
    aYLb (_float_): _aY anti-Lambda decay parameter_
    aZLb (_float_): _aZ anti-Lambda decay parameter_
    """
#  Print the values of class instance parameters:
    def printPar(self):
        """_Prints class instance parameters_
        """

        print("Parameters aPsi=%.3f"%self.aPsi,"pPsi=%.3f"%self.pPsi,"Pe=%.3f"%self.Pe, 
              "aXL=%.3f"%self.aXL,"aYL=%.3f"%self.aYL,"aZL=%.3f"%self.aZL,
              "aXLb=%.3f"%self.aXLb,"aYLb=%.3f"%self.aYLb,"aZLb=%.3f"%self.aZLb)
        
    def calc_pdf(self, tL, tp, pp, cp, tpb, ppb, cpb):
        """_This is an example of calculation of an probability distribution._
    # 
        Args:
            tL (_float_): _theta Lambda_
            tp (_float_): _theta proton_
            pp (_float_): _phi proton_
            cp (_float_): _chi proton_
            tpb (_float_): _theta anti-proton_
            ppb (_float_): _phi anti-proton_
            cpb (_float_): _chi anti-proton_
        """

        aPsi = self.aPsi
        pPsi = self.pPsi
        Pe = self.Pe
        aXL = self.aXL
        aYL = self.aYL
        aZL = self.aZL
        aXLb = self.aXLb
        aYLb = self.aYLb
        aZLb = self.aZLb

        # Little change
        # d = self.DL0.gen_2_other_based_on_1('ax', 0.3)
        d = self.DL0.get_params_file("t.txt")
        self.DL0.aX = d[0]
        self.DL0.aY = d[1]
        self.DL0.aZ = d[2]
        #
        # step, needed in derivative_calculator
        h = 0.0001

        # Derivative calculation interface for production
        # dericalcprod = derivative_calculator_prod()
        # vars_CC = np.array([aPsi, pPsi, Pe])
        
        # Derivative calculation interface for decay
        dericalc = derivative_calculator_decay()
        vars_DL0 = np.array([aXL, aYL, aZL])
        dericalc = derivative_calculator_decay()
        vars_DLb0 = np.array([aXLb, aYLb, aZLb])

        # derivatives for CC
        # dericalcprod.get_kinematic(tL)
        # deri_dC_aP = dericalcprod.calc_derivatives(vars_CC, 0, h)
        # deri_dC_pP = dericalcprod.calc_derivatives(vars_CC, 1, h)
        # deri_dC_Pe = dericalcprod.calc_derivatives(vars_CC, 2, h)
        
        # derivatives for DL0
        deri_dL0_aX = dericalc.calc_derivatives(vars_DL0, 0, h)
        deri_dL0_aY = dericalc.calc_derivatives(vars_DL0, 1, h)
        deri_dL0_aZ = dericalc.calc_derivatives(vars_DL0, 2, h)

        # derivatives for DLb0
        deri_dLb0_aX = dericalc.calc_derivatives(vars_DLb0, 0, h)
        deri_dLb0_aY = dericalc.calc_derivatives(vars_DLb0, 1, h)
        deri_dLb0_aZ = dericalc.calc_derivatives(vars_DLb0, 2, h)

        mat_start = time.time()
        # self.CC.get_kine_calc_trig(tL)

        # Rotation mattices
        rot_CC = RotationMatrix(theta=tL)
        rot_CC.create()
        r_CC = rot_CC.R

        rot_DL0 = RotationMatrix(theta=tp, phi=pp, chi=cp)
        rot_DL0.create()
        r_DL0 = rot_DL0.R

        rot_DLb0 = RotationMatrix(theta=tpb, phi=ppb, chi=cpb)
        rot_DLb0.create()
        r_DLb0 = rot_DLb0.R

        # Matrices before rotation (we don't rotate dC_A matrix)
        # dC_A = self.CC.create_matrix()
        dC_A = self.CC.C


        dL0_A = self.DL0.create_matrix()
        dLb0_A = self.DLb0.create_matrix()

        # Apply rotation
        dL0_A = np.dot(r_DL0, dL0_A)
        dLb0_A = np.dot(r_DLb0, dLb0_A)     

        dC_Pz = self.CC.dCPz
        # dC_aP = deri_dC_aP
        # dC_pP = deri_dC_pP
        # dC_Pe = deri_dC_Pe

        # Apply rotation to derivatives matrices
        dL0_aX = np.dot(r_DL0, deri_dL0_aX)
        dL0_aY = np.dot(r_DL0, deri_dL0_aY)
        dL0_aZ = np.dot(r_DL0, deri_dL0_aZ)
        dLb0_aX = np.dot(r_DLb0, deri_dLb0_aX)
        dLb0_aY = np.dot(r_DLb0, deri_dLb0_aY)
        dLb0_aZ = np.dot(r_DLb0, deri_dLb0_aZ)

        mat_end = time.time()

        self.mat_init = mat_end-mat_start
        pdf = 0

        start = time.time()
        pdf = multi_dot([dL0_A[:, 0], dC_A, dLb0_A[:, 0]])

        # ---------------------------------   Analytic derivatives calculation      

        dpz = multi_dot([dL0_A[:, 0], dC_Pz, dLb0_A[:, 0]])
        # dapsi = multi_dot([dL0_A[:, 0], dC_aP, dLb0_A[:, 0]]) 
        # dppsi = multi_dot([dL0_A[:, 0], dC_pP, dLb0_A[:, 0]])
        # dpe = multi_dot([dL0_A[:, 0], dC_Pe, dLb0_A[:, 0]])

        daxl = multi_dot([dL0_aX[:, 0], dC_A, dLb0_A[:, 0]])
        dayl = multi_dot([dL0_aY[:, 0], dC_A, dLb0_A[:, 0]])
        dazl = multi_dot([dL0_aZ[:, 0], dC_A, dLb0_A[:, 0]])
        daxlb = multi_dot([dL0_A[:, 0], dC_A, dLb0_aX[:, 0]])
        daylb = multi_dot([dL0_A[:, 0], dC_A, dLb0_aY[:, 0]])
        dazlb = multi_dot([dL0_A[:, 0], dC_A, dLb0_aZ[:, 0]])

        end = time.time()
        self.exec_time_loop_pdf += end-start

        self.dvec = [pdf, daxl, dayl, dazl, daxlb, daylb, dazlb, dpz]
        
        return [pdf, daxl, dayl, dazl, daxlb, daylb, dazlb, dpz]

    def gg(self,x):
        """_summary_
        This is the direct interface to communicate with the vegas package.
        This function is needed for vegas.Integrator()
        In this case it is working with 8-dimensional vector of parameters. 
        Args:
        x (_list_): _8-dimensional point_

        Returns:
        _class 'vegas._vegas.Integrator'_: _the flatten covariance matrix_
        """

        tL = x[0]
        tp = x[1]
        pp = x[2]
        cp = x[3]
        tpb = x[4]
        ppb = x[5]
        cpb = x[6]

        start = time.time()
        self.calc_pdf(tL, tp, pp, cp, tpb, ppb, cpb)

        dvec = self.dvec  # the vector of derivatives
        inv_pdf = 1/dvec[0]  # inverse of the PDF function

        jac = sin(tL)*sin(tp)*sin(tpb)  # Jacobian
        additional_row = np.asarray(dvec)*jac
        
        dvec_np = np.array(dvec[1:])
        res = np.outer(dvec_np, dvec_np)*jac*inv_pdf # the covariance matrix 

        res = res.flatten() # flattening 
        res = np.append(res, np.asarray(dvec)*jac)
        
        end = time.time()
        self.whole_gg = end-start

        return res    # the flatten covariance matrix

