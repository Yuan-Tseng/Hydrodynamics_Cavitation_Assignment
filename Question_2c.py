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
timestep = np.linspace(0, timespan[1], 1800)        # descretized time range from 0-0.1s
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

def lossless_prop_x():
    return (rho0 * c0**2) / (beta * Pa * k0)

def prop_x(Max_grad_time):
    return Max_grad_time * c0

def max_grad_time(gradient, t_all):
    Max_grad_idx = np.argmax(gradient)
    Max_grad_time = t_all[Max_grad_idx]
    return Max_grad_time
    

if __name__ == "__main__":
    sol = spi.solve_ivp(rhs, 
                  timespan, 
                  S0, 
                  t_eval = timestep, 
                  method='BDF')
    Amplitude = np.max(np.abs(sol.y * Pa/1000), axis=0)
    Gradient = np.max(np.abs(np.gradient(sol.y * Pa, dX, axis=0)), axis=0)
    t_all = sol.t
    Max_grad = np.max(Gradient)
    Max_grad_time = max_grad_time(Gradient, t_all)

    print(lossless_prop_x)
    print(prop_x(Max_grad_time))
    if lossless_prop_x() != prop_x(Max_grad_time):
        print("Shock formation distance does not match the lossless theoretical value.")
        print ("Lossless theoretical shock formation distance: {:.4f} m".format(lossless_prop_x()))
        print ("Numerical shock formation distance: {:.4f} m".format(prop_x(Max_grad_time)))
    else:
        print("Shock formation distance matches the lossless theoretical value.")
        print("Shock formation distance: {:.4f} m".format(lossless_prop_x()))

    Pa_values = np.linspace(75E3, 100E3, 3)
    f_values = np.linspace(1E6, 3E6, 8)

    # Create a 2D array to store shock distances: rows = Pa, cols = f
    shock_distance_matrix = np.zeros((len(Pa_values), len(f_values)))

    # Loop over pressure and frequency values
    for i, p in enumerate(Pa_values):
        for j, f in enumerate(f_values):
            k0 = 2 * np.pi * f / c0
            m = (beta / (rho0 * c0)) * p
            S0 = np.sin(k0 * X)

            def rhs_local(t, S):
                return BurgersEquation(t, S, X, m, d)

            sol = spi.solve_ivp(rhs_local, 
                                timespan, 
                                S0, 
                                t_eval=timestep, 
                                method='BDF')
            
            Gradient = np.max(np.abs(np.gradient(sol.y * p, dX, axis=0)), axis=0)
            shock_distance = prop_x(max_grad_time(Gradient, sol.t))
            shock_distance_matrix[i, j] = shock_distance

    # Plot shock distance vs frequency for each Pa
    plt.figure(figsize=(8, 5))
    for i, p in enumerate(Pa_values):
        plt.plot(f_values / 1e6,                # Convert to MHz
                shock_distance_matrix[i, :], 
                marker='o', 
                label=f"{p/1000:.0f} kPa")

    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Shock Formation Distance [m]")
    # plt.title("Shock Distance vs Frequency at Different Pa")
    plt.grid(True)
    plt.legend(title="Pressure Amplitude")
    plt.tight_layout()
    plt.show()