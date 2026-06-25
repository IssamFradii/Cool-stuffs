import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. SETUP 2D GRID (Expiries vs Tenors)
# ==========================================
np.random.seed(42)
expiries = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])     # Expiries from 3M to 10Y
tenors = np.array([1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0]) # Tenors from 1Y to 30Y

# Create Meshgrid for 3D processing and plotting
T_exp, T_ten = np.meshgrid(expiries, tenors, indexing='ij')

# Generate a synthetic SVI Volatility Surface 
svi_vol_matrix = np.zeros_like(T_exp)
vega_matrix = np.zeros_like(T_exp)

for i, exp in enumerate(expiries):
    for j, ten in enumerate(tenors):
        # Parametric generation mimicking realistic Level, Slope, Curvature
        level = 0.40 - 0.01 * exp
        slope_factor = (1 - np.exp(-0.3 * ten)) / (0.3 * ten)
        curve_factor = slope_factor - np.exp(-0.3 * ten)
        
        base_vol = level - 0.15 * slope_factor + 0.10 * curve_factor
        svi_vol_matrix[i, j] = base_vol
        vega_matrix[i, j] = np.sqrt(exp * ten) # Higher Vega for longer expiries/tenors

# Inject a clear outlier in the matrix to test the model's robustness
svi_vol_matrix[2, 3] += 0.08  # Anomaly at Expiry=1Y, Tenor=5Y

# ==========================================
# 2. HIERARCHICAL FACTOR EXTRACTION 
# ==========================================
def nss_curve(tau, b0, b1, b2, l1):
    """Simplified Nelson-Siegel curve for stable factor extraction."""
    tau = np.maximum(tau, 1e-5) # Prevent division by zero
    term1 = (1 - np.exp(-l1 * tau)) / (l1 * tau)
    term2 = term1 - np.exp(-l1 * tau)
    return b0 + b1 * term1 + b2 * term2

def loss_fn(params, tau, true_vol, vega):
    """Vega-weighted Least Squares Loss."""
    b0, b1, b2, l1 = params
    if l1 <= 0: return 1e10 # Strict bound for decay parameter
    pred_vol = nss_curve(tau, b0, b1, b2, l1)
    return np.sum(vega * (true_vol - pred_vol)**2)

# Lists to store the extracted factors (PCA equivalents)
b0_list, b1_list, b2_list = [], [], []
nss_vol_matrix = np.zeros_like(svi_vol_matrix)

for i, exp in enumerate(expiries):
    # Fit the term structure along the Tenor axis for THIS specific Expiry
    true_vols = svi_vol_matrix[i, :]
    vegas = vega_matrix[i, :]
    tenor_axis = tenors
    
    # Optimization with reasonable bounds to prevent instability
    res = minimize(
        loss_fn, 
        x0=[0.3, -0.1, 0.1, 0.5], 
        args=(tenor_axis, true_vols, vegas), 
        bounds=[(0.01, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.01, 5.0)], 
        method='L-BFGS-B'
    )
    
    b0, b1, b2, l1 = res.x
    b0_list.append(b0)
    b1_list.append(b1)
    b2_list.append(b2)
    
    # Reconstruct the fitted curve for this expiry row
    nss_vol_matrix[i, :] = nss_curve(tenor_axis, b0, b1, b2, l1)

# ==========================================
# 3. VISUALIZATION
# ==========================================
fig = plt.figure(figsize=(18, 6))

# Plot 1: The Extracted Factors over Expiries (PCA-like)
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(expiries, b0_list, marker='o', label='Hierarchy 0: Level (β0)', color='blue', linewidth=2)
ax1.plot(expiries, b1_list, marker='s', label='Hierarchy 1: Slope (β1)', color='orange', linewidth=2)
ax1.plot(expiries, b2_list, marker='^', label='Hierarchy 2: Curvature (β2)', color='green', linewidth=2)
ax1.set_title('Extracted Volatility Factors across Expiries\n(Like PCA Components)', fontsize=12)
ax1.set_xlabel('Expiries (Years)')
ax1.set_ylabel('Factor Loading / Value')
ax1.grid(True, alpha=0.4)
ax1.legend()

# Plot 2: 3D Surface of Original SVI Matrix
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
surf1 = ax2.plot_surface(T_exp, T_ten, svi_vol_matrix, cmap='viridis', alpha=0.8, edgecolor='k', linewidth=0.5)
ax2.scatter([expiries[2]], [tenors[3]], [svi_vol_matrix[2, 3]], color='red', s=100, label='Outlier (1Yx5Y)')
ax2.set_title('Original SVI Surface (Expiries vs Tenors)\nRaw Matrix with Anomaly', fontsize=12)
ax2.set_xlabel('Expiry')
ax2.set_ylabel('Tenor')
ax2.set_zlabel('Implied Volatility')
ax2.legend()

# Plot 3: 3D Surface of Fitted NSS Matrix
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
surf2 = ax3.plot_surface(T_exp, T_ten, nss_vol_matrix, cmap='plasma', alpha=0.8, edgecolor='k', linewidth=0.5)
ax3.set_title('Fitted Hierarchical Parameter Surface\n(Smooth & No-Arbitrage)', fontsize=12)
ax3.set_xlabel('Expiry')
ax3.set_ylabel('Tenor')
ax3.set_zlabel('Implied Volatility')

plt.tight_layout()
plt.show()
