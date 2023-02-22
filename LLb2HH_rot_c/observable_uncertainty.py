import numpy as np
from uncertainties import ufloat
import uncertainties
from math import *

def calc_uncertainty(alpha, sigma_alpha):
    """_summary_
        This function calcluates the uncertainty of the CP violating observable A_particle_name
    Args:
        alpha (_float_): _decay parameter_
        sigma_alpha (_float_): _standard deviation of the alpha parameter_

    Returns:
        _float_: _standard deviation of the given observable_
    """
    alpha_bar = -alpha
    sigma_alpha_bar = sigma_alpha
    fraction_bar = (2*alpha_bar*sigma_alpha) / ((alpha-alpha_bar)**2)
    fraction = (2*alpha*sigma_alpha_bar) / ((alpha-alpha_bar)**2)

    result = np.sqrt(fraction_bar**2 + fraction**2)
    return result


def calc_observable(alpha):
    """_summary_
        This function calcluates the value of the CP violating observable A_particle_name
    Args:
        alpha (_float_): _decay parameter_
    Returns:
        _float_: _the value of the given observable_
    """
    alpha_bar = -alpha
    A = (alpha+alpha_bar) / (alpha-alpha_bar)
    return A

def calc_variance_sum(al, sigma_al, aXi, sigma_aXi, covariance):
    alpha, axi = uncertainties.correlated_values([al, aXi], covariance)
    alpha_bar, axibar = uncertainties.correlated_values([-al, -aXi], covariance)

    print("----------------------------")
    print("covariance")
    print(covariance)

    A = (alpha+alpha_bar) / (alpha-alpha_bar)
    print(A)

    Axi = (axi+axibar) / (axi-axibar)
    print(Axi)

    Asum = A + Axi
    print(Asum)

    cov_matrix = uncertainties.covariance_matrix([A, Axi, Asum])
    ar_cov_matrix = np.asarray(cov_matrix)
    print(ar_cov_matrix)

    var_A_l = ar_cov_matrix[0][0]
    var_A_Xi = ar_cov_matrix[1][1]

    var_Asum = var_A_l + 2*ar_cov_matrix[0][1] + var_A_Xi

    print("The value of A_l + A_Xi, Var(A_l + A_Xi), sigma(A_l + A_Xi)")
    print(Asum, var_Asum, sqrt(var_Asum))

    return np.array([A, Axi, Asum, var_Asum])
