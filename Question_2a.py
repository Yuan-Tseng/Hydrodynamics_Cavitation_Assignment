import numpy as np
import scipy.integrate as spi
import matplotlib as plt

def BurgersEquation(t, S, X, m, d):
    #Wave number discretization
    N_x = X.size
    dx = X[1] - X[0]
    k = 2*np.pi*np.fft.fftfreq(N_x, d = dx)

    #Spatial derivative in the Fourier domain
    S_hat = np.fft.fft(S)
    S_hat_x = 1j*k*S_hat
    S_hat_xx = -k**2*S_hat

    # Back to spatial domain
    S_x = np.fft.ifft(S_hat_x)
    S_xx = np.fft.ifft(S_hat_xx)

    # Compute Burgers equation time derivative
    S_t = -m*S*S_x + d*S_xx

    # Ensure real output
    S_t = S_t.real
    return S_t

