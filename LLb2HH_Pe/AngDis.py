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
    Pe = 0.5
    aL  = 0.758
    aLb = -0.758
    exec_time_loop_pdf = 0
    whole_gg = 0
    mat_init = 0


    def __init__(self, PaPsi=aPsi, PpPsi=pPsi, PPe=Pe):
        self.aPsi = PaPsi
        self.pPsi = PpPsi
        self.Pe = PPe
        self.DL0 = AADecay12(self.aL)
        self.DLb0 = AADecay12(self.aLb)
        self.CC = AAProd12(PaPsi,PpPsi,PPe)
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

    # @property
    # def DL0(self):
    #     return AADecay12(self.aL)
    
    # @property
    # def DLb0(self):
    #     return AADecay12(self.aLb)

    # @property
    # def CC(self):
    #     return AAProd12(self.aPsi, self.pPsi, self.Pe)

#  Print the values of class instance parameters:
    def printPar(self):
        """_Prints class instance parameters_
        """

        print("Parameters aPsi=%.3f"%self.aPsi,"pPsi=%.3f"%self.pPsi,"Pe=%.3f"%self.Pe, 
              "aL=%.3f"%self.aL,"aLb=%.3f"%self.aLb)
    
    def calc_pdf(self, tL, tp, pp, tpb, ppb):
        """_This is an example of calculation of an probability distribution._
    # 
        Args:
            tL (_float_): _theta Lambda_
            tp (_float_): _theta proton_
            pp (_float_): _phi proton_
            tpb (_float_): _theta anti-proton_
            ppb (_float_): _phi anti-proton_
        """

        aPsi = self.aPsi
        pPsi = self.pPsi
        Pe = self.Pe
        aL = self.aL
        aLb = self.aLb
        
        #dC = self.CC.MatrixD(tL)
        #dCa = self.CC.dCa
        #dCp = self.CC.dCp

        
        # step, needed in derivative_calculator
        h = 0.0001

        # Derivative calculation interface for production
        dericalcprod = derivative_calculator_prod()
        vars_CC = np.array([aPsi, pPsi, Pe])
        
        # Derivative calculation interface for decay
        dericalc = derivative_calculator_decay()
        vars_DL0 = np.array([aL, 0])
        vars_DLb0 = np.array([aLb, 0])

        # derivatives for CC
        dericalcprod.get_kinematic(tL)
        deri_dC_aP = dericalcprod.calc_derivatives(vars_CC, 0, h)
        deri_dC_pP = dericalcprod.calc_derivatives(vars_CC, 1, h)
        deri_dC_Pe = dericalcprod.calc_derivatives(vars_CC, 2, h)
        
        # derivatives for DL0
        dericalc.get_kinematic(tp, pp)
        deri_dL0_aM = dericalc.calc_derivatives(vars_DL0, 0, h)
        deri_dL0_pM = dericalc.calc_derivatives(vars_DL0, 1, h)

        # derivatives for DLb0
        dericalc.get_kinematic(tpb, ppb)
        deri_dLb0_aM = dericalc.calc_derivatives(vars_DLb0, 0, h)
        deri_dLb0_pM = dericalc.calc_derivatives(vars_DLb0, 1, h)

        mat_start = time.time()
        self.CC.get_kine_calc_trig(tL)
        self.DL0.get_kine_calc_trig(tp, pp)
        self.DLb0.get_kine_calc_trig(tpb, ppb)

        dC_A = self.CC.create_matrix()
        dL0_A = self.DL0.create_matrix()
        dLb0_A = self.DLb0.create_matrix()

        dC_aP = deri_dC_aP
        dC_pP = deri_dC_pP
        dC_Pe = deri_dC_Pe
        dL0_aM = deri_dL0_aM
        dL0_pM = deri_dL0_pM
        dLb0_aM = deri_dLb0_aM
        dLb0_pM = deri_dLb0_pM

        mat_end = time.time()

        self.mat_init = mat_end-mat_start
        pdf = 0

        start = time.time()
        V = 2*(4*pi)**2*(3+aPsi)/3.0  # volume
        V1 = -2*(4*pi)**2*(3+aPsi)*(3+aPsi)/3.0  # volume derivative
        pdf = multi_dot([dL0_A[:,0], dC_A, dLb0_A[:,0]])/V

        # ---------------------------------   Analytic derivatives calculation      
        
        dapsi = multi_dot([dL0_A[:,0], dC_aP, dLb0_A[:,0]])/V + pdf*V/V1
        dppsi = multi_dot([dL0_A[:,0], dC_pP, dLb0_A[:,0]])/V
        dpe = multi_dot([dL0_A[:,0], dC_Pe, dLb0_A[:,0]])/V
        dal = multi_dot([dL0_aM[:,0], dC_A, dLb0_A[:,0]])/V
        dalb = multi_dot([dL0_A[:,0], dC_A, dLb0_aM[:,0]])/V
        
        end = time.time()
        self.exec_time_loop_pdf += end-start

        self.dvec = [pdf, dal, dalb, dapsi, dppsi, dpe]

        return [pdf, dal, dalb, dapsi, dppsi, dpe]

    def gg(self,x):
        """_summary_
        This is the direct interface to communicate with the vegas package.
        This function is needed for vegas.Integrator()
        In this case it is working with 5-dimensional vector of parameters. 
        Args:
        x (_list_): _5-dimensional point_

        Returns:
        _class 'vegas._vegas.Integrator'_: _the flatten covariance matrix_
        """

        tL = x[0]
        tp = x[1]
        pp = x[2]
        tpb = x[3]
        ppb = x[4]

        start = time.time()
        self.calc_pdf(tL, tp, pp, tpb, ppb)

        dvec = self.dvec  # the vector of derivatives
        inv_pdf = 1/dvec[0]  # inverse of the PDF function

        dvec_np = np.array(dvec[1:])

        jac = sin(tL)*sin(tp)*sin(tpb)  # Jacobian
        res = np.outer(dvec_np, dvec_np)*jac*inv_pdf # the covariance matrix

        newres = res[:5]
        newres2 = newres[:, :5]
        
        end = time.time()
        self.whole_gg = end-start

        #return res.flatten()    # the flatten covariance matrix
        return newres2.flatten()    # the flatten covariance matrix

