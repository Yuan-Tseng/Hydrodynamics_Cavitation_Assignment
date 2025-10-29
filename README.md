# Hydrodynamics and Cavitation Numerical Assignment

# Hydrodynamics and Cavitation – Numerical Assignment (ETH Zürich)

This repository contains the full implementation of the numerical assignment for the course *Hydrodynamics and Cavitation (151-0125-00L)* at ETH Zürich.

The assignment explores three classical nonlinear or unstable flow phenomena using numerical simulations and Python-based visualization:

### 1. **Capillary Bridge Between Two Loops**
A liquid bridge suspended between two circular loops is analyzed using a parametric profile governed by a cosh-like solution. The stability of two possible solutions is investigated based on surface area minimization, and the variation of neck radius and contact angle with respect to loop separation is evaluated. The critical separation distance beyond which no stable bridge can form is also computed.

### 2. **Nonlinear Acoustic Wave Propagation (Burgers' Equation)**
Using the simplified Westervelt equation transformed into a viscous Burgers' equation, we simulate the temporal evolution of high-amplitude acoustic waves in 1D. This part investigates:
- Wave steepening and shock formation
- Amplitude and gradient evolution in time (log–log plots)
- Shock formation distance vs. theoretical lossless predictions
- Parametric sweep over frequency and amplitude

### 3. **Rayleigh–Plateau Instability**
The breakup of a liquid column into droplets is explored via linear stability analysis. The most unstable wavenumber is identified, and pressure/radius perturbation fields are visualized.

## 🎯 Key Physics Concepts

- Surface tension and capillary stability
- Shock wave formation in nonlinear acoustics
- Linear stability and surface-mode breakup of jets