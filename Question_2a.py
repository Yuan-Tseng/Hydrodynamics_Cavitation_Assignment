import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

# Physical Parameters
f0 = 1E6        # frequency
Pa = 100E3      # pressure Amplitude (Reference)
rho0 = 1000     # water density [kg*m^-3]
c0 = 1500       # speed of sound in water [m/s]
beta = 3.5      # non-linearity coefficient
delta = 4.3E-6  # sound diffusivity

# Derived Parameters
Lambda = c0 / f0                        # wave length [m]
k0 = 2 * np.pi / Lambda                 # wavenumber
m = (beta / (rho0 * c0)) * Pa           # non dimensionalized m by mutiplying Pa
d = delta / 2                           # non dimensionalized diffusivity

# Space and Time
timespan = [0, 0.1]                                 # total time span [s]
timestep = np.linspace(0, timespan[1], 1000)        # descretized time range from 0-0.1s
NX = 512                                            # number of spatial points  
X = np.linspace(0, Lambda, NX, endpoint=False)      # spatial domain from 0 to lambda
dX = X[1] - X[0]                                    # spatial step size

# Initial Condition
S0 = np.sin(k0 * X)

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

def rhs(t, S):
    return BurgersEquation(t, S, X, m, d)

if __name__ == "__main__":
    sol = spi.solve_ivp(rhs, 
                  timespan, 
                  S0, 
                  t_eval = timestep, 
                  method='BDF')
    # print(sol)
    indices = [0, 8, 16, 50, 500]  # Different time spot
    fig, axes = plt.subplots(len(indices), 1, figsize=(8, 8), sharex=True)

    for i, idx in enumerate(indices):
        t = sol.t[idx]*1000
        S_profile = sol.y[:, idx]

        ax = axes[i]
        ax.plot(X / Lambda, S_profile, color='b')
        ax.set_ylabel("S")
        ax.set_title(f"Time = {t:.1f} ms")
        ax.grid(True)

    axes[-1].set_xlabel("x / λ", fontsize=11) 
    plt.tight_layout()
    plt.show()