import numpy as np
from math import *
from RotMat import *
class AADecay12new:
    def __init__(self, aX, aY, aZ):
        self.aX = aX
        self.aY = aY
        self.aZ = aZ
        self.A = np.zeros((4, 4))
        self.A[0, 0] = 1
    
    def gen_2_other_based_on_1(var_name='aX', var_value=0.):
        """_This function generates 2 parameters based on the input parameter value in a way that
            the following condition is fulfilled:

            aX^2 + aY^2 + aZ^2 = 1_

            Through the var_name parameter we specify the parameter on the basis of which the others will be generated.

        Args:
            var_name (str, optional): _name of the chosen variable_. Defaults to 'aX'.
            var_value (_float_, optional): _value of the chosen variable_. Defaults to 0..
        
        Returns:
            _numpy.ndarray_: one-dimensional vector of parameters [aX, aY, aZ]__
        """

        match var_name:
            case 'aX':
                input_var = var_value
                val = 1 - input_var** 2
                other_var_gen_1 = np.random.uniform(0, val)
                val2 = val - other_var_gen_1 ** 2
                other_var_gen_2 = np.sqrt(val - other_var_gen_1 ** 2)
                
                return np.array([input_var, other_var_gen_1, other_var_gen_2])

            case 'aY':
                input_var = var_value
                val = 1 - input_var** 2
                other_var_gen_1 = np.random.uniform(0, val)
                val2 = val - other_var_gen_1 ** 2
                other_var_gen_2 = np.sqrt(val - other_var_gen_1 ** 2)
                
                return np.array([other_var_gen_2, input_var, other_var_gen_1])

            case 'aZ':
                input_var = var_value
                val = 1 - input_var** 2
                other_var_gen_1 = np.random.uniform(0, val)
                val2 = val - other_var_gen_1 ** 2
                other_var_gen_2 = np.sqrt(val - other_var_gen_1 ** 2)
                
                return np.array([other_var_gen_2, other_var_gen_1, input_var])
            
            case _:
                print("Wrong variable name!")
    
    def create_matrix(self):
        """_This functions creates a decay matrix. Martix elements
        are filled via class instance atributes._

        Returns:
            _ndarray_: _a decay matrix_
        """

        aX = self.aX
        aY = self.aY
        aZ = self.aZ

        self.A[1, 0] = aX
        self.A[2, 0] = aY
        self.A[3, 0] = aZ

        return self.A

