import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ==========================================
# 1. SETUP SYNTHETIC EURIBOR & DELTA DATA
# ==========================================
# Pillar Maturities (in years)
pillars = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])

# Synthetic Euribor Rates (Normal upward sloping curve, e.g., 2% to 3.5%)
euribor_rates = 0.02 + 0.015 * (1 - np.exp(-0.2 * pillars)) 

# Synthetic Pillar Deltas (e.g., PV01 risk in Euros per basis point)
# Let's say the desk is heavily exposed to the 5Y and 10Y pillars
pillar_deltas = np.array([500, 1200, -800, 300, 15000, 2000, -12000, 1000, 500, -200])

# ==========================================
# 2. FIT THE MACRO EURIBOR CURVE (NSS)
# ==========================================
def nss_rate(T, b0, b1, b2, b3, l1, l2):
    """Calculates the NSS rate for maturity T."""
    T = np.maximum(T, 1e-5)
    term1 = (1 - np.exp(-l1 * T)) / (l1 * T)
    term2 = term1 - np.exp(-l1 * T)
    term3 = (1 - np.exp(-l2 * T)) / (l2 * T) - np.exp(-l2 * T)
    return b0 + b1 * term1 + b2 * term2 + b3 * term3

def nss_loss(params, T, true_rates):
    b0, b1, b2, b3, l1, l2 = params
    if l1 <= 0 or l2 <= 0 or l1 == l2: return 1e10 # Strict constraints
    pred_rates = nss_rate(T, b0, b1, b2, b3, l1, l2)
    return np.sum((true_rates - pred_rates)**2)

# Calibrate the curve
res = minimize(
    nss_loss, 
    x0=[0.03, -0.01, 0.01, 0.01, 0.5, 0.1], 
    args=(pillars, euribor_rates),
    bounds=[(0, 0.1), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (0.01, 5.0), (0.01, 5.0)],
    method='L-BFGS-B'
)
b0, b1, b2, b3, l1, l2 = res.x

# ==========================================
# 3. CALCULATE THE JACOBIAN VECTORS (SENSITIVITIES)
# ==========================================
# These are the analytical derivatives of the NSS function with respect to the Beta parameters
def get_jacobian(T, l1, l2):
    T = np.maximum(T, 1e-5)
    dB0 = np.ones_like(T)                                      # Level Vector
    dB1 = (1 - np.exp(-l1 * T)) / (l1 * T)                     # Slope Vector
    dB2 = dB1 - np.exp(-l1 * T)                                # Curvature 1 Vector
    dB3 = ((1 - np.exp(-l2 * T)) / (l2 * T)) - np.exp(-l2 * T) # Curvature 2 Vector
    
    # Return as an N x 4 Matrix
    return np.column_stack((dB0, dB1, dB2, dB3))

# Calculate the N x 4 matrix for our specific pillars
jacobian_matrix = get_jacobian(pillars, l1, l2)

# ==========================================
# 4. HIERARCHICAL DELTA PROJECTION
# ==========================================
# Matrix multiplication: (1 x N) Deltas @ (N x 4) Jacobian
macro_deltas = pillar_deltas @ jacobian_matrix

delta_b0, delta_b1, delta_b2, delta_b3 = macro_deltas

print("\n=== MACRO RISK COMPRESSION COMPLETED ===")
print(f"Level Risk (Beta 0):      €{delta_b0:,.2f} per bp")
print(f"Slope Risk (Beta 1):      €{delta_b1:,.2f} per bp")
print(f"Curvature 1 Risk (Beta 2): €{delta_b2:,.2f} per bp")
print(f"Curvature 2 Risk (Beta 3): €{delta_b3:,.2f} per bp")

# ==========================================
# 5. VISUALIZATION PLOTS
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot A: The Hierarchical Jacobian Vectors
T_dense = np.linspace(0.1, 30, 100)
dense_jacobian = get_jacobian(T_dense, l1, l2)

axes[0].plot(T_dense, dense_jacobian[:, 0], label='Level Sensitivity ($\\beta_0$)', lw=2, color='blue')
axes[0].plot(T_dense, dense_jacobian[:, 1], label='Slope Sensitivity ($\\beta_1$)', lw=2, color='orange')
axes[0].plot(T_dense, dense_jacobian[:, 2], label='Curvature 1 Sensitivity ($\\beta_2$)', lw=2, color='green')
axes[0].plot(T_dense, dense_jacobian[:, 3], label='Curvature 2 Sensitivity ($\\beta_3$)', lw=2, color='red')
axes[0].axhline(0, color='black', lw=1, ls='--')
axes[0].set_title("The Hierarchical Vectors (Jacobian Components)", fontsize=12)
axes[0].set_xlabel("Maturity (Years)")
axes[0].set_ylabel("Sensitivity (Multiplier)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot B: The Risk Compression (Original vs Macro)
categories = ['Total Pillar Risk (Raw)', 'Level ($\\beta_0$)', 'Slope ($\\beta_1$)', 'Curv 1 ($\\beta_2$)', 'Curv 2 ($\\beta_3$)']
values = [np.sum(pillar_deltas), delta_b0, delta_b1, delta_b2, delta_b3]
colors = ['gray', 'blue', 'orange', 'green', 'red']

axes[1].bar(categories, values, color=colors, alpha=0.7)
axes[1].axhline(0, color='black', lw=1)
axes[1].set_title("Risk Compression: Bucket Deltas to Macro Factors", fontsize=12)
axes[1].set_ylabel("Euro Risk per Basis Point (PV01)")
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
