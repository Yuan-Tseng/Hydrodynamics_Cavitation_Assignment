# Question 3a - Rayleigh-Plateau Instability - 
# Plot capillary timescale growth coefficient vs dimentionless wavenumber
# Show max growth rate and corresponding wavenumber

import numpy as np
from scipy.special import iv
import matplotlib.pyplot as plt

density = 1000          # Density of water (kg/m^3)
surface_tension = 0.072 # Surface tension of water (N/m)
R0 = 20e-3              # Radius of the jet (m)
jet_velocity = 1.0      # Velocity of the jet (m/s)
k_list = np.linspace(0/R0, 1.5/R0, 500) # Dimentionless wavenumber values

def capillary_time():
    '''Capillary timescale'''
    return np.sqrt(density * R0**3 / surface_tension)

def dispersion_omega(k):
    '''Dispersion relation ω(k)'''
    omega_sq = (-1 / capillary_time()**2) * ((k*R0 * iv(1, k*R0)) / (iv(0, k*R0)) * (1 - (k*R0)**2))
    omega = np.lib.scimath.sqrt(omega_sq)
    return omega

if __name__ == "__main__":
    kr = k_list * R0                # dimensionless wavenumber k * R0
    growth_rate = np.imag(dispersion_omega(k_list) * capillary_time()) # imaginary part of dimensionless growth rate ω * t_cap
    
    mask = np.isfinite(growth_rate) # filter out non-finite values

    kr_plot = kr[mask]              # filtered dimensionless wavenumber
    growth_plot = growth_rate[mask] # filtered dimensionless growth rate

    max = np.nanargmax(growth_plot) # index of max growth rate
    kr_max = kr_plot[max]           # corresponding dimensionless wavenumber
    growth_max = growth_plot[max]   # max growth rate

    print (f'Max growth rate = {growth_max:.4f} at kR_0 = {kr_max:.4f}')

    plt.figure(figsize=(8,5))
    plt.plot(kr_plot, growth_plot, lw=2, color='b')
    plt.plot(kr_max, growth_max, 'ro')
    plt.axvline(kr_max, color='r', linestyle='--', alpha=0.6)

    plt.xlabel(r'$k R_0$', fontsize=13)
    plt.ylabel(r'$\tau \, \mathrm{Im}(\omega)$', fontsize=13)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()