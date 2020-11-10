import numpy as np
from numpy import fft

# Some utilities have their own files, include them here
from .fftshow import fftshow
from .homomorphic_filter import homomorphic_filter

def nonzero(x):
    """
    If given 0 then this returns an extremely tiny, but non-zero, positive value. Otherwise the
    given value is returned. The value x must be a scalar and cannot be an array.
    """
    return np.finfo(float).eps if x == 0 else x

def psf2otf(psf, shape):
    """
    Convert a PSF to an OTF. This is essentially np.fft.fft2(psf, shape) except it also includes a
    minor shift of the data so that the center of the PSF is at (0,0) before the Fourier transform
    is computed but after padding.
    """
    psf_shape = psf.shape
    psf = np.pad(psf, ((0, shape[0] - psf_shape[0]), (0, shape[1] - psf_shape[1])), 'constant')
    psf = np.roll(psf, (-(psf_shape[0]//2), -(psf_shape[1]//2)), axis=(0,1)) # shift PSF so center is at (0,0)
    return fft.fft2(psf)

def otf2psf(otf, shape):
    """
    Convert an OTF to a PSF. This is essentially np.fft.ifft2(otf, shape) except it also includes a
    minor shift of the data so that the center of the PSF is moved back to the middle after the
    inverse Fourier transform is computed but before cropping.
    """
    psf = fft.ifft2(otf)
    psf = np.roll(psf, (shape[0]//2, shape[1]//2), axis=(0,1)) # shift PSF so center is in middle
    return psf[:shape[0], :shape[1]]

def __x2y2(w, h):
    """Creates the mesh grid of x^2 + y^2 from -w/2 to w/2 and -h/2 to h/2"""
    x,y = np.ogrid[-(w//2):((w+1)//2), -(h//2):((h+1)//2)]
    return y*y + x*x

def ideal_low_pass(w, h, D):
    """
    Creates a Fourier-space ideal low-pass filter of the given width and height with the cutoff D.
    """
    return (__x2y2(w,h)<=(D*D)).astype(float)

def ideal_high_pass(w, h, D):
    """
    Creates a Fourier-space ideal high-pass filter of the given width and height with the cutoff D.
    """
    return (__x2y2(w,h)>(D*D)).astype(float)

def butterworth_low_pass(w, h, D, n):
    """
    Creates a Fourier-space Butterworth low-pass filter of the given width and height with the
    cutoff D and order n.
    """
    return 1 / (1 + (__x2y2(w,h)/(D*D))**n)

def butterworth_high_pass(w, h, D, n):
    """
    Creates a Fourier-space Butterworth high-pass filter of the given width and height with the
    cutoff D and order n.
    """
    return 1 - butterworth_low_pass(w, h, D, n)

def gaussian(w, h, sigma, normed=False):
    """
    Creates a Gaussian centered in a w x h image with the given standard deviation sigma. By
    default this has a peak of 1. If normed is True then this is normalized so it sums to 1.
    """
    g = np.exp(-__x2y2(w,h)/(sigma*sigma))
    if normed: g /= g.sum()
    return g

def gaussian_low_pass(w, h, sigma):
    """
    Creates a Gaussian low-pass filter centered in a w x h image with the given standard deviation
    sigma.
    """
    return gaussian(w, h, sigma)

def gaussian_high_pass(w, h, sigma):
    """
    Creates a Gaussian high-pass filter centered in a w x h image with the given standard deviation
    sigma.
    """
    return 1 - gaussian(w, h, sigma)
