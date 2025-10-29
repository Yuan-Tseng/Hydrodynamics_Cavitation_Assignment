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
timestep = np.linspace(0, timespan[1], 2500)        # descretized time range from 0-0.1s
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
    Amplitude = np.max(np.abs(sol.y * Pa/1000), axis=0)
    Gradient = np.max(np.abs(np.gradient(sol.y * Pa, dX, axis=0)), axis=0)
    t_all = sol.t*1000
    Max_grad = np.max(Gradient)
    Max_grad_idx = np.argmax(Gradient)
    Max_grad_time = t_all[Max_grad_idx]

    # === Plot Amplitude and Gradient in subplots ===
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Subplot 1: Amplitude vs Time
    axes[0].plot(t_all, Amplitude, color='purple')
    axes[0].set_ylabel("max |p(x, t)| [kPa]")
    # axes[0].set_title("Maximum Pressure Amplitude vs Time")
    axes[0].grid(True, which='both', linestyle='--', alpha=0.6)

    # Subplot 2: Gradient vs Time
    axes[1].plot(t_all, Gradient, color='darkorange')
    axes[1].set_ylabel("max |∂p/∂x| [Pa/m]")
    axes[1].set_xlabel("Time [ms] (log scale)")
    # axes[1].set_title("Maximum Pressure Gradient vs Time")
    axes[1].grid(True, which='both', linestyle='--', alpha=0.6)
    axes[1].axvline(Max_grad_time, color='gray', linestyle='--', linewidth=1)

    # Annotate maximum gradient point
    label_text = f"t = {Max_grad_time:.3f} ms"
    axes[1].text(Max_grad_time, Max_grad * 0.9,
                label_text, rotation=90, color='black',
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

    # Set log scale on time axis
    axes[0].set_xscale('log')
    axes[1].set_xscale('log')

    plt.tight_layout()
    plt.show()