import numpy as np
import scipy.integrate as spi
import matplotlib as plt

f0 = 1E6        # frequency
Pa = 100E3      # pressure Amplitude (Reference)
density = 1000  # water density [kg*m^-3]
c0 = 1500       # speed of sound in water [m/s]
beta = 3.5      # non-linearity coefficient
delta = 4.3E-6  # sound diffusivity
timestep = np.linspace(0, 0.1, 500)     # descretized time range from 0-0.1s

def BurgersEquation(t, S, X, m, d):
    '''Input in spatial, solved in spectral, transform again to spatial as output'''
    '''The output is an array of S at timestep t'''
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

if __name__ == "main":
    print("Super")

