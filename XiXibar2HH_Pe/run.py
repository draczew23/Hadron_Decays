import numpy as np
import os
# import main.py
# Limits
start_point = 1
# end_point = 1

# Here we pass values of our arguments in one list 
params_vals = np.array([start_point])

# Conversion to string. Converted list of parameters will be passed as command
params_to_command = " ".join([str(i) for i in params_vals])

# Step to change parameters
step = 0.05

# Iterations number
# itera = np.arange(start_point, end_point, step)
# N = len(itera)

# The file we want to run with parameters
runfile = 'main.py'

for i in range(1):
    os.system('python3 {} {}'.format(runfile, params_to_command))
    params_vals = params_vals + step
    params_to_command = " ".join([str(j) for j in params_vals])

