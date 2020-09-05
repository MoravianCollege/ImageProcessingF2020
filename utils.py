import numpy as np

# Some utilities have their own files, include them here
from .fftshow import fftshow
from .homomorphic_filter import homomorphic_filter

def nonzero(x):
    """
    If given 0 then this returns an extremely tiny, but non-zero, positive value. Otherwise the
    given value is returned. The value x must be a scalar and cannot be an array.
    """
    return np.finfo(float).eps if x == 0 else x
