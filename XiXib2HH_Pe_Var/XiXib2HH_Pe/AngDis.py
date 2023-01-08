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
    aPsi = 0.586
    pPsi = 1.213
    Pe = 0.
    aX  = -0.376 # Xi- -> Lambda pi-
    pX  = 0.011 # Xi- -> Lambda pi-
    aXb  = 0.376 # Xi+ -> Lambda pi+
    pXb  = -0.011 # Xi+ -> Lambda pi+
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
        self.DX = AADecay12(self.aX,self.pX)
        self.DXb = AADecay12(self.aXb,self.pXb)
        self.CC = AAProd12(PaPsi,PpPsi,PPe)
    """
    An example class of probability distributions and interface to vegas
#
    Attributes:
    aPsi (_float_): _alpha Jpsi decay parameter_
    pPsi (_float_): _phi Jpsi relative phase_
    Pe (_float_): _polarization electron beam_
    aX (_float_): _alpha Xi- decay parameter_
    pX (_float_): _phi Xi- decay parameter_
    aXb (_float_): _alpha Xi+ decay parameter_
    pXb (_float_): _phi Xi+ decay parameter_
    aL (_float_): _alpha Lambda decay parameter_
    aLb (_float_): _alpha anti-Lambda decay parameter_
    """
#  Print the values of class instance parameters:
    def printPar(self):
        """_Prints class instance parameters_
        """

        print("Parameters aPsi=%.3f"%self.aPsi,"pPsi=%.3f"%self.pPsi,"Pe=%.3f"%self.Pe, 
              "aX=%.3f"%self.aX,"pX=%.3f"%self.pX,
              "aXb=%.3f"%self.aXb,"pXb=%.3f"%self.pXb,
              "aL=%.3f"%self.aL,"aLb=%.3f"%self.aLb)
        
    def calc_pdf(self, tX, tL, pL, tp, pp, tLb, pLb, tpb, ppb):
        """_This is an example of calculation of an probability distribution._
    # 
        Args:
            tX (_float_): _theta Xi-_
            tL (_float_): _theta Lambda_
            pL (_float_): _phi Lambda_
            tp (_float_): _theta proton_
            pp (_float_): _phi proton_
            tLb (_float_): _theta anti-Lambda_
            pLb (_float_): _phi anti-Lambda_
            tpb (_float_): _theta anti-proton_
            ppb (_float_): _phi anti-proton_
        """

        aPsi = self.aPsi
        pPsi = self.pPsi
        Pe = self.Pe
        aX = self.aX
        pX = self.pX
        aXb = self.aXb
        pXb = self.pXb
        aL = self.aL
        aLb = self.aLb
        
        # step, needed in derivative_calculator
        h = 0.001

        # Derivative calculation interface for production
        dericalcprod = derivative_calculator_prod()
        vars_CC = np.array([aPsi, pPsi, Pe])
        
        # Derivative calculation interface for decay
        dericalc = derivative_calculator_decay()
        vars_DX = np.array([aX, pX])
        vars_DXb = np.array([aXb, pXb])
        vars_DL0 = np.array([aL, 0])
        vars_DLb0 = np.array([aLb, 0])

        # derivatives for CC
        dericalcprod.get_kinematic(tX)
        deri_dC_aP = dericalcprod.calc_derivatives(vars_CC, 0, h)
        deri_dC_pP = dericalcprod.calc_derivatives(vars_CC, 1, h)
        deri_dC_Pe = dericalcprod.calc_derivatives(vars_CC, 2, h)
        
        # derivatives for DX
        dericalc.get_kinematic(tL, pL)
        deri_dX_aM = dericalc.calc_derivatives(vars_DX, 0, h)
        deri_dX_pM = dericalc.calc_derivatives(vars_DX, 1, h)

        # derivatives for DXb
        dericalc.get_kinematic(tLb, pLb)
        deri_dXb_aM = dericalc.calc_derivatives(vars_DXb, 0, h)
        deri_dXb_pM = dericalc.calc_derivatives(vars_DXb, 1, h)

        # derivatives for DL0
        dericalc.get_kinematic(tp, pp)
        deri_dL0_aM = dericalc.calc_derivatives(vars_DL0, 0, h)
        deri_dL0_pM = dericalc.calc_derivatives(vars_DL0, 1, h)

        # derivatives for DLb0
        dericalc.get_kinematic(tpb, ppb)
        deri_dLb0_aM = dericalc.calc_derivatives(vars_DLb0, 0, h)
        deri_dLb0_pM = dericalc.calc_derivatives(vars_DLb0, 1, h)

        mat_start = time.time()
        self.CC.get_kine_calc_trig(tX)
        self.DX.get_kine_calc_trig(tL, pL)
        self.DXb.get_kine_calc_trig(tLb, pLb)
        self.DL0.get_kine_calc_trig(tp, pp)
        self.DLb0.get_kine_calc_trig(tpb, ppb)

        dC_A = self.CC.create_matrix()
        dX_A = self.DX.create_matrix()
        dXb_A = self.DXb.create_matrix()
        dL0_A = self.DL0.create_matrix()
        dLb0_A = self.DLb0.create_matrix()

        dC_aP = deri_dC_aP
        dC_pP = deri_dC_pP
        dC_Pe = deri_dC_Pe
        dX_aM = deri_dX_aM
        dX_pM = deri_dX_pM
        dXb_aM = deri_dXb_aM
        dXb_pM = deri_dXb_pM
        dL0_aM = deri_dL0_aM
        dL0_pM = deri_dL0_pM
        dLb0_aM = deri_dLb0_aM
        dLb0_pM = deri_dLb0_pM

        mat_end = time.time()

        self.mat_init = mat_end-mat_start
        pdf = 0

        start = time.time()
        # V = 2*(4*pi)**4*(3+aPsi)/3.0  # volume
        # V1 = -2*(4*pi)**4*(3+aPsi)*(3+aPsi)/3.0  # volume derivative
        # #pdf = multi_dot([dL0_A[:,0], dX_A, dC_A, dXb_A, dLb0_A[:,0]])
        # pdf = multi_dot([dL0_A[:,0], dC_A, dX_A, dXb_A, dLb0_A[:,0]])

        # # ---------------------------------   Analytic derivatives calculation      
        
        # #dapsi = multi_dot([dL0_A[:,0], dX_A, dC_aP, dXb_A, dLb0_A[:,0]])
        # #dppsi = multi_dot([dL0_A[:,0], dX_A, dC_pP, dXb_A, dLb0_A[:,0]])
        # #dpe = multi_dot([dL0_A[:,0], dX_A, dC_Pe, dXb_A, dLb0_A[:,0]])
        # #dal = multi_dot([dL0_aM[:,0], dX_A, dC_A, dXb_A, dLb0_A[:,0]])
        # #dax = multi_dot([dL0_A[:,0], dX_aM, dC_A, dXb_A, dLb0_A[:,0]])
        # #dpx = multi_dot([dL0_A[:,0], dX_pM, dC_A, dXb_A, dLb0_A[:,0]])
        # #dalb = multi_dot([dL0_A[:,0], dX_A, dC_A, dXb_A, dLb0_aM[:,0]])
        # #daxb = multi_dot([dL0_A[:,0], dX_A, dC_A, dXb_aM, dLb0_A[:,0]])
        # #dpxb = multi_dot([dL0_A[:,0], dX_A, dC_A, dXb_pM, dLb0_A[:,0]])
        # dapsi = multi_dot([dL0_A[:,0], dC_aP, dX_A, dXb_A, dLb0_A[:,0]])
        # dppsi = multi_dot([dL0_A[:,0], dC_pP, dX_A, dXb_A, dLb0_A[:,0]] )
        # dpe = multi_dot([dL0_A[:,0], dC_Pe, dX_A,  dXb_A, dLb0_A[:,0]])
        # dal = multi_dot([dL0_aM[:,0], dC_A, dX_A, dXb_A, dLb0_A[:,0]])
        # dax = multi_dot([dL0_A[:,0], dC_A, dX_aM, dXb_A, dLb0_A[:,0]] )
        # dpx = multi_dot([dL0_A[:,0], dC_A, dX_pM, dXb_A, dLb0_A[:,0]] )
        # dalb = multi_dot([dL0_A[:,0], dC_A, dX_A, dXb_A, dLb0_aM[:,0]])
        # daxb = multi_dot([dL0_A[:,0], dC_A, dX_A, dXb_aM, dLb0_A[:,0]])
        # dpxb = multi_dot([dL0_A[:,0], dC_A, dX_A, dXb_pM, dLb0_A[:,0]])
        
        pdf_x = np.dot(dX_A, dL0_A[:,0])  #-> define only full Xi decay     
        pdf_xb = np.dot(dXb_A, dLb0_A[:,0])  #-> define only full Xibar decay  
        pdf = multi_dot([pdf_x, dC_A, pdf_xb])  #-> final product of all matrices

        # The same for the derivatives:
        dapsi = multi_dot([pdf_x, dC_aP, pdf_xb])           
        dppsi = multi_dot([pdf_x, dC_pP, pdf_xb] )          
        dpe = multi_dot([pdf_x, dC_Pe, pdf_xb])             
        dal_x = np.dot(dX_A, dL0_aM[:,0])                   
        dal = multi_dot([dal_x, dC_A, pdf_xb])              
        dax_x = np.dot(dX_aM, dL0_A[:,0])                   
        dax = multi_dot([dax_x, dC_A, pdf_xb] )             
        dpx_x = np.dot(dX_pM, dL0_A[:,0] )                  
        dpx = multi_dot([dpx_x, dC_A, pdf_xb] )             
        dalb_xb = np.dot(dXb_A, dLb0_aM[:,0])               
        dalb = multi_dot([pdf_x, dC_A, dalb_xb])            
        daxb_xb = np.dot(dXb_aM, dLb0_A[:,0])               
        daxb = multi_dot([pdf_x, dC_A, daxb_xb])            
        dpxb_xb = np.dot(dXb_pM, dLb0_A[:,0])               
        dpxb = multi_dot([pdf_x, dC_A, dpxb_xb])            


        end = time.time()
        self.exec_time_loop_pdf += end-start

        self.dvec = [pdf, dal, dalb, dax, dpx, daxb, dpxb, dapsi, dppsi, dpe]

        return [pdf, dal, dalb, dax, dpx, daxb, dpxb, dapsi, dppsi, dpe]

    def gg(self,x):
        """_summary_
        This is the direct interface to communicate with the vegas package.
        This function is needed for vegas.Integrator()
        In this case it is working with 9-dimensional vector of parameters. 
        Args:
        x (_list_): _9-dimensional point_

        Returns:
        _class 'vegas._vegas.Integrator'_: _the flatten covariance matrix_
        """

        tX = x[0]
        tL = x[1]
        pL = x[2]
        tp = x[3]
        pp = x[4]
        tLb = x[5]
        pLb = x[6]
        tpb = x[7]
        ppb = x[8]

        start = time.time()
        self.calc_pdf(tX, tL, pL, tp, pp, tLb, pLb, tpb, ppb)

        dvec = self.dvec  # the vector of derivatives
        inv_pdf = 1/dvec[0]  # inverse of the PDF function

        jac = sin(tX)*sin(tL)*sin(tp)*sin(tLb)*sin(tpb)  # Jacobian
        additional_row = np.asarray(dvec)*jac

        dvec_np = np.array(dvec[1:])
        res = np.outer(dvec_np, dvec_np)*jac*inv_pdf # the covariance matrix

        res = res.flatten()  # flattening
        res = np.append(res, np.asarray(dvec)*jac)
        
        end = time.time()
        self.whole_gg = end-start

        return res    # the flatten covariance matrix

