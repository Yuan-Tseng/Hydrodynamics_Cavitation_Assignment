# Question 1b - Capillary Bridge - Show the existence of H_critical 
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo

R = 1.0                             # Radius of the loop 
H_vals = np.linspace(0.01, 2, 1000) # H values to test, from 0.01 to 2
K1_list = []                        # Store K1 values
K2_list = []                        # Store K2 values
H_valid = []                        # Store valid H values where two solutions exist

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

if __name__ == "__main__":
    K1_list, K2_list, H_valid = K_solution()
    H_crit = H_valid[-1]
    print(f"Critical H = {H_crit:.6f}")

    # Plotting the results
    plt.plot(H_valid[5:], K1_list[5:], label="K1 (unstable)")
    plt.plot(H_valid, K2_list, label="K2 (stable)")
    plt.axvline(H_crit, color='r', linestyle='--', label=f"H_crit = {H_crit:.2f}")
    plt.xlabel("H")
    plt.ylabel("K values")
    plt.legend()
    plt.grid(True)
    plt.show()