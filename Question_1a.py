# Question 1a - Capillary Bridge - Show why the larger K value is the stable solution
import numpy as np
import scipy.integrate as spi
import scipy.optimize as spo

R = 1.0  # Radius of the loop 
H = 1.0  # Height of the capillary bridge

def equation_K(K):
    """Equation to solve for K given R and H"""
    return K * np.cosh(H/(2*K)) - R

def K_solution():
    """Solve for K using the given R and H"""
    K0_1 = 0.01     # Initial guess (smaller value)
    K0_2 = 100      # Another initial guess (larger value)
    K_sol_1 = spo.fsolve(equation_K, K0_1)
    K_sol_2 = spo.fsolve(equation_K, K0_2)
    return K_sol_1[0], K_sol_2[0]

def radius(z):
    """Calculate the radius of the capillary bridge at height z"""
    K1, K2 = K_solution()
    K = max(K1, K2) # Choose the larger K for the stable solution
    return K * np.cosh(z / K)

def contact_angle():
    """Calculate the contact angle (beta) at the loop"""
    K1, K2 = K_solution()
    K = max(K1, K2) # Choose the larger K for the stable solution
    beta = np.pi/2 - np.arctan(np.sinh(H/(2*K)))
    return beta

def area(K):
    """Calculate the surface area of the capillary bridge"""
    integrand = lambda z: K * np.cosh(z/K) * np.sqrt(1 + (np.sinh(z/K))**2)
    S, _ = spi.quad(integrand, -H/2, H/2) # Integrate from -H/2 to H/2
    return 2 * np.pi * S

if __name__ == "__main__":
    K1, K2 = K_solution()
    print("K1 value:", K1, "  K2 value:", K2)       #Question 1a
    print("Surface Area with K1:", area(K1))        #Question 1a
    print("Surface Area with K2:", area(K2))        #Question 1a