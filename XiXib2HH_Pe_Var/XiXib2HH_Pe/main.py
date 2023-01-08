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
import sys

start = time.time() # start of execution
# particle masses 
PI = np.pi
# Access to PDF namespace
pdf=PDF(PPe=1)
# Change parameters for given class instance:
#pdf.Pz=0.6
if len(sys.argv) != 1:
    pdf.Pe = float(sys.argv[1])
else:       # to check if directly is ok
    pdf.Pe = 1

pdf.printPar()
print(pdf.aPsi, pdf.pPsi, pdf.Pe)
# Define function to communicate with vegas:
gg=pdf.gg
# # Integrator creation
integ = vegas.Integrator([[0, PI], [0, PI], [-PI, PI], [0, PI], [-PI, PI], [0, PI], [-PI, PI], [0, PI], [-PI, PI]])

# # # Training
class_time = time.time()

training = integ(gg, nitn=10, neval=2000)
result = integ(gg, nitn=20, neval=10000)

add_row = result[-10:]
result = result[:-10]

arr = np.array([[ufloat(x.mean, x.sdev)] for x in result]) # change dtype to float64

norm_row = np.array([[ufloat(x.mean, x.sdev)] for x in add_row]) # change dtype to float64
norm_row = norm_row.flatten()

# Repair printing
print("Additional row ".format(norm_row))
print("---------------------------------")

# Create dimensions based on the dimension of the result

# Create the matrix of our values
A = arr.reshape((9, 9))
A_norm = np.array([ufloat(0, 0) for x in range(len(arr))]) # covariance matrix
A_norm = A_norm.reshape(A.shape)

# Normalization loop
for i in range(9):
    for j in range(9):
        A_norm[i, j] = A[i, j]/norm_row[0] - norm_row[i+1]*norm_row[j+1]/(norm_row[0]**2)

A1 = unumpy.matrix(A_norm)
B = A1.I    # invert matrix

C = np.array([ufloat(0, 0) for x in range(len(arr))]) # covariance matrix
C = C.reshape(B.shape)

for k in range(0, 9):
    for l in range(k, 9):
        C[k,l] = B[k, l]/unumpy.sqrt(B[k, k])/unumpy.sqrt(B[l, l])

print("Correlation matrix for parameters:")
print("  aL","        aLb","          aXi","           pXi","          aXib","           pXib","          aPsi","           pPsi","           Pe")
print(C)

correlation_aL_aLb = C[0, 1]

diagB = B.diagonal()
diagBsqrt = unumpy.sqrt(diagB)

di = np.diag_indices(9)     # diagonal indices of 8-dimensional matrix
C[di] =  diagBsqrt          # put sqrt values into the main diagonal

print("standard deviations for the global parameters:")
print("s(aL)= ", '{:.1u}'.format(C[0,0])," s(aLb)= ", '{:.1u}'.format(C[1,1]),
      " s(aXi)= ", '{:.1u}'.format(C[2,2])," s(pXi)= ", '{:.1u}'.format(C[3,3]),
      " s(aXib)= ", '{:.1u}'.format(C[4,4])," s(pXib)= ", '{:.1u}'.format(C[5,5]),
      " s(aPsi)= ", '{:.1u}'.format(C[6,6])," s(pPsi)= ", '{:.1u}'.format(C[7,7]),
      " s(Pe)= ", '{:.1u}'.format(C[8,8]))

end = time.time()

# # Saving output result
# scanned_parameter = float(pdf.Pe)
# C_vals = [C[0,0], C[1,1], C[2,2], C[3,3], C[4,4], C[5,5], C[6,6], C[7,7], C[8,8]]

# with open('output/result_scanPe.txt', 'a') as f:
#     f.write('{:.3f};'.format(scanned_parameter))
#     for val in C_vals:
#         f.write('{:.3f};'.format(val.nominal_value))
#         f.write('{:.3f};'.format(val.std_dev))

#     f.write('\n')
# f.close()    

# vals = '{:.2f}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}'.format(float(pdf.Pe), C[0,0], C[1,1], C[2,2], C[3,3], C[4,4], C[5,5], C[6,6], C[7,7], C[8,8])

# # Save values of arguments in a txt file
# f = open('output/scanPe.txt','a')
# a = vals
# f.writelines(vals + "\n")
# f.close()

print("The time of execution of above program is :", end-start)
