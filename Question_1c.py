# Question 1c - Capillary Bridge - Plot neck radius vs H and contact angle vs H
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

R = 1.0                             # Radius of the loop 
H_vals = np.linspace(0.01, 2, 1000) # H values to test, from 0.01 to 2
K1_list = []                        # Store K1 values
K2_list = []                        # Store K2 values
H_valid = []                        # Store valid H values where two solutions exist
beta_list = []                      # Store contact angle values
neck_radius_list = []               # Store neck radius values

def equation_K(K, H):
    """Equation to solve for K given R and H"""
    return K * np.cosh(H/(2*K)) - R

def K_solution():
    """Solve for K using the given R and H"""
    for H in H_vals:
        try:
            # Solve for K with two different initial guesses
            K0_1 = 0.0025
            K0_2 = 2.0
            K1 = spo.fsolve(equation_K, K0_1, args=(H, ))[0]
            K2 = spo.fsolve(equation_K, K0_2, args=(H, ))[0]

            # when K1 and K2 converge, we reach the critical height
            err = abs(K1-K2)
            if err < 5e-4 and H > 0.1:  # the setting of H > 0.1 is to avoid K1 = K2 at very small H
                break                   # H critical found

            # Store valid solutions
            K1_list.append(K1)
            K2_list.append(K2)
            H_valid.append(H)

        except Exception:
            continue

    return K1_list, K2_list, H_valid

def contact_angle(K2_list, H_valid):
    """Calculate the contact angle (beta) at the loop"""
    for H, K in zip(H_valid, K2_list):
        beta = np.pi/2 - np.arctan(np.sinh(H / (2*K)))
        beta_list.append(beta)
    return beta_list

def neck_radius(K2_list):
    """Calculate the neck radius of the capillary bridge"""
    neck_radius_list = K2_list # at z=0, since cosh(0)=1
    return neck_radius_list

if __name__ == "__main__":
    K1_list, K2_list, H_valid = K_solution()
    beta_list = contact_angle(K2_list, H_valid)
    r_neck_list = neck_radius(K2_list)
    
    # --- Plotting in subplots ---
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Subplot 1: Neck radius
    axs[0].plot(H_valid, r_neck_list, color='blue', label="Neck radius $r(0)$")
    axs[0].set_ylabel("Neck radius $r(0)$")
    axs[0].legend()
    axs[0].grid(True)

    # Subplot 2: Contact angle
    axs[1].plot(H_valid, beta_list, color='green', label="Contact angle $\\beta$")
    axs[1].set_xlabel("H")
    axs[1].set_ylabel("Contact angle $\\beta$ (rad)")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()