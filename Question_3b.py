# Question 3b - Rayleigh-Plateau Instability - 
# Plot radial deformation R = R0 + R'
# Plot pressure variation field p'(xref, r)

import numpy as np
from scipy.special import iv
import matplotlib.pyplot as plt

density = 1000          # Density of water (kg/m^3)
surface_tension = 0.072 # Surface tension of water (N/m)
R0 = 20e-3              # Radius of the jet (m)
jet_velocity = 1.0      # Velocity of the jet (m/s)
k = 0.69739479/R0       # Dimentionless wavenumber
h = 2e-3                # Initial perturbation amplitude (m)
t = 0.0                 # Time (s)

x_ref = np.linspace(0, 0.5, 500)  # Reference x positions along the jet

def capillary_time():
    '''Capillary timescale'''
    return np.sqrt(density * R0**3 / surface_tension)

def dispersion_omega():
    '''Dispersion relation ω(k) (imaginary part only)'''
    omega_sq = (-1 / capillary_time()**2) * ((k*R0 * iv(1, k*R0)) / (iv(0, k*R0)) * (1 - (k*R0)**2))
    omega = np.lib.scimath.sqrt(omega_sq)
    return omega

def radial_deformation():
    '''Radial deformation R = R0 + R' at reference positions x_ref and time t'''
    R_prime = np.real(h * np.exp(1j * ((k - dispersion_omega()/jet_velocity)*x_ref - k*jet_velocity*t)))
    return R0 + R_prime.real

def pressure_variation():
    '''Pressure variation field p'(xref, r) at time t'''
    omega = dispersion_omega()
    phi = (k - omega / jet_velocity) * x_ref - k * jet_velocity * t  # shape of phase at x_ref

    R_profile = radial_deformation() # shape of interface along x_ref
    r_max = np.max(R_profile)        
    r = np.linspace(0, r_max, 1000)  # create r grid up to max radius

    # Meshgrid for final 2D field
    p_prime_grid = np.full((len(x_ref), len(r)), np.nan)

    # Boundary condition 
    pR = surface_tension * (k**2 * h - h / R0**2)
    A = pR / iv(0, k * R0)

    for i, R_now in enumerate(R_profile):
        r_mask = r <= R_now  # mask for points inside the jet at this x_ref
        p_hat_r = A * iv(0, k * r[r_mask])
        p_prime_grid[i, r_mask] = np.real(p_hat_r * np.exp(1j * phi[i]))

    return r, p_prime_grid

if __name__ == "__main__":
    r, p_prime = pressure_variation()

    # Plot
    # Note: 
    # Radial pressure variation is not obvious (smaller r has smaller aplitude p_hat_r)
    # Only axial direction is clear
    plt.figure(figsize=(10, 4))
    contour = plt.contourf(x_ref*100, r*1000, p_prime.T, 100, cmap='coolwarm')  # transpose for (r, x)
    plt.colorbar(contour, label='Pressure [Pa]')
    plt.xlabel(r'$x_{ref}$ [cm]', fontsize=13)
    plt.ylabel(r'$radius$ [mm]', fontsize=13)
    # plt.title("Pressure perturbation $p'(x_{ref, r)$ at $t=0$")
    plt.ylim(0, 25)
    plt.tight_layout()
    plt.plot(x_ref*100, radial_deformation()*1000, 'k-', label='Jet radius $R(x)$')
    plt.grid()
    plt.legend()
    plt.show()