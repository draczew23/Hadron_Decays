import numpy as np
import vegas
from math import *
from sympy import var
import gvar as gv

from AngDis import PDF
import time
from uncertainties import unumpy
from uncertainties import ufloat
import observable_uncertainty

start = time.time() # start of execution
# particle masses 
PI = np.pi
# Access to PDF namespace
pdf=PDF()
# Change parameters for given class instance:
#pdf.aLb=0.6
pdf.printPar()
print(pdf.aPsi, pdf.pPsi, pdf.Pe)
# Define function to communicate with vegas:
gg=pdf.gg
# # Integrator creation
integ = vegas.Integrator([[0, PI], [0, PI], [-PI, PI], [0, PI], [-PI, PI], [0, PI], [-PI, PI]])

# # # Training
class_time = time.time()

training = integ(gg, nitn=10, neval=2000)
result = integ(gg, nitn=20, neval=10000)

class_time_end = time.time()

#print(result)
arr = np.array([[ufloat(x.mean, x.sdev)] for x in result]) # change dtype to float64
#print(arr)

# Create dimensions based on the dimension of the result

# Create the matrix of our values
A = arr.reshape((7, 7))
A = unumpy.matrix(A)
B = A.I    # invert matrix

Bprim = B.A

row1 = Bprim[0]
row2 = Bprim[2]

v = np.array([row1[0].nominal_value, row1[2].nominal_value, row2[0].nominal_value, row2[2].nominal_value])
v = np.reshape(v, (2, 2))

print("-------------")
print(Bprim)

C = np.array([ufloat(0, 0) for x in range(len(arr))]) # covariance matrix
C = C.reshape(B.shape)

for k in range(0, 7):
    for l in range(k, 7):
        C[k,l] = B[k, l]/unumpy.sqrt(B[k, k])/unumpy.sqrt(B[l, l])

print("Correlation matrix for parameters:")
print("  aL","        aLb","          aXi","           pXi","          aPsi","           pPsi","           Pe")
print(C)

#correlation_aL_aLb = C[0, 1]

diagB = B.diagonal()
diagBsqrt = unumpy.sqrt(diagB)

di = np.diag_indices(7)     # diagonal indices of 6-dimensional matrix
C[di] =  diagBsqrt          # put sqrt values into the main diagonal

print("standard deviations for the global parameters:")
print("s(aL)= ", '{:.1u}'.format(C[0,0])," s(aLb)= ", '{:.1u}'.format(C[1,1]),
      " s(aXi)= ", '{:.1u}'.format(C[2,2])," s(pXi)= ", '{:.1u}'.format(C[3,3]),
      " s(aPsi)= ", '{:.1u}'.format(C[4,4])," s(pPsi)= ", '{:.1u}'.format(C[5,5]),
      " s(Pe)= ", '{:.1u}'.format(C[6,6]))
print("correlations for the global parameters:")
print("aL/aLb= ", '{:.1u}'.format(C[0,1]),"aL/aXi= ", '{:.1u}'.format(C[0,2]),
      "aLb/aXi= ", '{:.1u}'.format(C[1,2]))

end = time.time()

print("The time of execution of above program is :", end-start)
