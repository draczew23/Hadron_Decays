import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import re

# get data files
# os.chdir(r'/home/user/NCBJ_files/NCBJ_main/TensorFlowAnalysis-master/work/Lc2pKpi')     # put your path
my_files = glob.glob('*.npz')

# load data and create keys
our_data = np.load('full_data.npz')
keys = our_data.keys()
key_names = []

for key in keys:
    key_names.append(key)

arlen = len(our_data[key_names[0]])

Adat = np.zeros(arlen)
Adat = np.reshape(Adat, (-1, 1))

# combine data from npz file into one array
for key in key_names:
    ar = our_data[key]
    Adat = np.column_stack((Adat, ar))

# Result
Adat = Adat[:, 1:]
# print(Adat.shape)

i=10

# Create complex amplitudes
amp1 = Adat[:, 2] + 1j * Adat[:, 3]
amp2 = Adat[:, 4] + 1j * Adat[:, 5]
amp3 = Adat[:, 6] + 1j * Adat[:, 7]
amp4 = Adat[:, 8] + 1j * Adat[:, 9]

# print(amp1.real)

# amp1 --
# amp2 +-
# amp3 -+
# amp4 ++

# b_rho_0 calculation
b_rho_0_0 = abs(amp1)**2 + abs(amp2)**2 + abs(amp3)**2 + abs(amp4)**2
b_rho_0_x = 2*np.real(amp2*np.conj(amp1) + amp4*np.conj(amp3))
b_rho_0_y = 2*np.imag(amp2*np.conj(amp1) + amp4*np.conj(amp3))
b_rho_0_z = -abs(amp1)**2 + abs(amp4)**2 + abs(amp2)**2 - abs(amp3)**2

b_rho_0 = np.array([b_rho_0_0, b_rho_0_x, b_rho_0_y, b_rho_0_z])

# b_rho_1 calculation
b_rho_1_0 = 2*np.real(amp4*np.conj(amp2) + amp1*np.conj(amp3))
b_rho_1_x = 2*np.real(amp4*np.conj(amp1) + amp3*np.conj(amp2))
b_rho_1_y = 2*np.imag(amp4*np.conj(amp1) + amp3*np.conj(amp2))
b_rho_1_z = 2*np.real(amp4*np.conj(amp2) - amp1*np.conj(amp3))

b_rho_1 = np.array([b_rho_1_0, b_rho_1_x, b_rho_1_y, b_rho_1_z])

# b_rho_2 calculation
b_rho_2_0 = 2*np.imag(amp2*np.conj(amp4) + amp1*np.conj(amp3))
b_rho_2_x = 2*np.imag(amp1*np.conj(amp4) - amp3*np.conj(amp2))
b_rho_2_y = 2*np.real(amp1*np.conj(amp4) - amp3*np.conj(amp2))
b_rho_2_z = 2*np.imag(amp2*np.conj(amp4) - amp1*np.conj(amp3))

b_rho_2 = np.array([b_rho_2_0, b_rho_2_x, b_rho_2_y, b_rho_2_z])

# b_rho_3 calculation
b_rho_3_0 = -abs(amp1)**2 - abs(amp2)**2 + abs(amp3)**2 + abs(amp4)**2
b_rho_3_x = 2*np.real(amp4*np.conj(amp3) - amp1*np.conj(amp2))
b_rho_3_y = 2*np.imag(amp4*np.conj(amp3) + amp1*np.conj(amp2))
b_rho_3_z = abs(amp1)**2 - abs(amp4)**2 - abs(amp2)**2 - abs(amp3)**2

b_rho_3 = np.array([b_rho_3_0, b_rho_3_x, b_rho_3_y, b_rho_3_z])

# full b array
b = np.array([b_rho_0, b_rho_1, b_rho_2, b_rho_3])

# extract one iteration of the b matrix
b_fragment = b[:, :, 0]
ar_to_file = np.zeros((4, 4))

# filling our array
ar_to_file[0, :] = b_fragment[0, :]
ar_to_file[:, 0] = b_fragment[:, 0]

np.savetxt('test.txt', ar_to_file, delimiter=';')
# masses
masses = Adat[:, :2]
