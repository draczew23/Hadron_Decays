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
pdf=PDF()
# Change parameters for given class instance:
pdf.printPar()
print(pdf.aPsi, pdf.pPsi, pdf.Pe)
# Define function to communicate with vegas:
gg=pdf.gg
# # Integrator creation
integ = vegas.Integrator([[0, PI], [0, PI], [-PI, PI], [-PI, PI], [0, PI], [-PI, PI], [-PI, PI]])

# # # Training
class_time = time.time()

training = integ(gg, nitn=10, neval=20)
result = integ(gg, nitn=20, neval=100)

add_row = result[-10:]
result = result[:-10]

arr = np.array([[ufloat(x.mean, x.sdev)] for x in result]) # change dtype to float64

norm_row = np.array([[ufloat(x.mean, x.sdev)] for x in add_row])
norm_row = norm_row.flatten()

# Repair printing
print("Additional row ".format(norm_row))
print("----------------------------")

# # Create dimensions based on the dimension of the result
# # Create the matrix of our values
A = arr.reshape((9, 9))
A_norm = np.array([ufloat(0, 0) for x in range(len(arr))]) # covariance matrix
A_norm = A_norm.reshape(A.shape)

# Normalization loop
for i in range(9):
        for j in range(9):
                A_norm[i, j] = A[i, j]/norm_row[0] -norm_row[i+1]*norm_row[j+1]/(norm_row[0]**2)

A1 = unumpy.matrix(A_norm)
B = A1.I    # invert matrix

C = np.array([ufloat(0, 0) for x in range(len(arr))]) # covariance matrix
C = C.reshape(B.shape)

for k in range(0, 9):
    for l in range(k, 9):
        C[k,l] = B[k, l]/unumpy.sqrt(B[k, k])/unumpy.sqrt(B[l, l])

print("Correlation matrix for parameters:")
#print("  gavL","         gwL","        aLb","          aPsi","           pPsi","             Pe")
print(C)

correlation_gavL_gwL = C[0, 1]

diagB = B.diagonal()
diagBsqrt = unumpy.sqrt(diagB)

di = np.diag_indices(9)     # diagonal indices of 5 dimensional matrix
C[di] =  diagBsqrt          # put sqrt values into the main diagonal

print("standard deviations for the global parameters:")
print("s(aXL)= ", '{:.1u}'.format(C[0,0]),"s(aYL)= ", '{:.1u}'.format(C[1,1]),"s(aZL)= ", '{:.1u}'.format(C[2,2]),
      "s(aXLb)= ", '{:.1u}'.format(C[3,3]),"s(aYLb)= ", '{:.1u}'.format(C[4,4]),"s(aZLb)= ", '{:.1u}'.format(C[5,5]),
        " s(aPsi)= ", '{:.1u}'.format(C[6,6])," s(pPsi)= ", '{:.1u}'.format(C[7,7]),
        " s(Pe)= ", '{:.1u}'.format(C[8,8]))


#print("correlations for the global parameters:")
#print("gavL/gwL= ", '{:.1u}'.format(C[0,1]),"gavL/aLb= ", '{:.1u}'.format(C[0,2]),
#      "gwL/aLb= ", '{:.1u}'.format(C[1,2]))

end = time.time()

# Saving output result
#scanned_parameter = float(pdf.gavL)
#C_vals = [C[0,0], C[1,1], C[2,2], C[3,3], C[4,4], C[5,5]]
#
#with open('output/result_scanaLb_new.txt', 'a') as f:
#    f.write('{:.2f};'.format(scanned_parameter))
#    for val in C_vals:
#        f.write('{:.3f};'.format(val.nominal_value))
#        f.write('{:.3f};'.format(val.std_dev))
#
#    f.write('\n')
#f.close()    
#
#vals = '{:.2f}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}'.format(float(pdf.aLb), C[0,0], C[1,1], C[2,2], C[3,3], C[4,4], C[5,5])
#
## Save values of arguments in a txt file
#f = open('output/scanaLb_new.txt','a')
#a = vals
#f.writelines(vals + "\n")
#f.close()
#
#vals_corr = '{:.2f}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}, {:.1u}'.format(float(pdf.aLb), C[0,1], C[0,2], C[0,3], C[0,4], C[0,5], C[1,2], C[1,3], C[1,4], C[1,5], C[2,3], C[2,4], C[2,5], C[3,4], C[3,5], C[4,5] )
#f = open('output/scanaLb_corr_new.txt','a')
#a = vals_corr
#f.writelines(vals_corr + "\n")
#f.close()

print("The time of execution of above program is :", end-start)
