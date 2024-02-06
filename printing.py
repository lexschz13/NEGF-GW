import numpy as np
from numba import njit


@njit
def float_string(x, decimals=1):
    assert decimals > 0
    
    s = str(int(x)) + '.'
    for i in range(decimals):
        x *= 10
        if i == decimals-1:
            s += str(round(x % 10))
        else:
            s += str(int(x % 10))
    return s


@njit
def exp_string(x, decimals=1):
    assert decimals > 0
    
    exp = int(np.log10(x))
    if exp >= 0:
        x /= 10**exp
    else:
        x *= 10**(-exp+1)
    s = float_string(x, decimals)
    if exp >= 0:
        s += "e+" + str(exp)
    else:
        s += "e"  + str(exp-1)
    
    return s