#import numpy as np
import numpy as np  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sympy as sp
import os

import pandas as pd
import matplotlib.tri as tri
from typing import Tuple
import warnings
from skimage import measure

from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

import ternary

# -------------------------- timers for calcs -----------------------
import time
_t0 = time.perf_counter()          # sets t = 0 for *this* run

def stamp(msg=""):
    """Prints   [elapsed-seconds]  message"""
    print(f"[{time.perf_counter() - _t0:8.3f}] {msg}")

def reset_stamp():
    """Reset the zero-point so the next stamp starts from 0.000"""
    global _t0
    _t0 = time.perf_counter()
# ------------------------------------------------------------------

"""
TERNARY MIXING IN SUPERCRITICAL MELTS
    Inputs at line 44 below
"""

warnings.filterwarnings("ignore", category=RuntimeWarning)

print('')
print('____________________ New Ternary Mixing Model ___________________')

# ============= INPUTS for Ternary grid and G surface ============
T    = 3000.
PGPa = 12.0
wtpercent_H2 = 4.8 # For subneptune



N = 100 # G surface resolution, 100, 30?
grid_N=110 # number of grid points for convex point detection, 160


L0_AB = 1.2*115000 # non-ideal mixing for Fe-H2, 115000, 1.2x the 115000 value matches solubility at low T
L1_AB = -9500  # Pressure effect on Fe-H2, -9500, required for [H2]_metal > [H2]_silicate
L0_BA = 17000 # non-ideal mixing for Fe-H2, 17000
L1_BA = -9500  # Pressure effect on Fe-H2, -95000

L_BC = -4950.0 # MgSiO3 - H2
L_CB = 622000.0 # MgSiO3 - H2
tau_BC = 4350 # New!!!!
ppi_BC = -35

#L_CA = 240000 - 26.0 * T + 390 * PGPa # MgSiO3 - Fe, 200000 - 25.0 * T + 390 * PGPa
#L_AC = 240000 - 26.0 * T + 390 * PGPa
L_CA = 240000 - 28.0 * T + 1116.2 * PGPa # MgSiO3-Fe new fit
L_AC = 240000 - 28.0 * T + 1116.2 * PGPa

L_ABC = 0
R = 8.314

# Use this to test sensitivity for mixing parameters
L_AC = L_AC*1.0 # MgSiO3-Fe total
L_CA = L_AC

L1_AB = L1_AB*1.0 # Pressure effect on Fe-H2

ppi_BC = ppi_BC*1.0 # Pressure effect on MgSiO3-H2


# Set list of interaction parameters
L_and_P = [L0_AB, L1_AB, L0_BA, L1_BA, L_BC, L_CB, L_CA, L_AC, tau_BC, ppi_BC, L_ABC, PGPa]

#============================ Gibbs Free Energy of Mixing ==================================
def gibbs_ternary_subregular(xA, xB, xC, T, R, *L_and_P):
    L0_AB, L1_AB, L0_BA, L1_BA, L_BC, L_CB, L_CA, L_AC, tau_BC, ppi_BC, L_ABC,  P = L_and_P
    """
    A = Fe, 
    B = H2, 
    C = silicate/oxide
    xA = mole fraction Fe
    xB = mole fraction H2
    xC = mole fraction silicate
    
    Utilizes solvus for silicate-H2 from Gilmore and Stixrude (2025),
    solvus for Mg-silicate-Fe metal from Insixiengmay and Stixrude (2024),
    and calibrated mixing along the Fe-H join.
    
    This formulation of the ternary that includers projection of mole fractions
    onto the binary joins is due to Ganguly (2001, EMU notes in Min v. 3, eq. 67).
    The xij terms in braces are equivalent to xij = xi/(xi+xj), and thus
    renormalized mole fractions for use with the binary interaction parameters. 
    """
    eps = 1e-12
    xA = max(xA, eps)
    xB = max(xB, eps)
    xC = max(xC, eps)
    total = xA + xB + xC
    xA /= total
    xB /= total
    xC /= total

    factor_BC = (1 - T / tau_BC + P / ppi_BC)
    factor_AC = 1.0

    G_ideal = R * T * (xA * np.log(xA) + xB * np.log(xB) + xC * np.log(xC))

    L_AB = L0_AB + L1_AB * P
    L_BA = L0_BA + L1_BA * P

    G_excess = (
        xA * xB * (L_AB * 0.5 * (1 + xB - xA) + L_BA * 0.5 * (1 + xA - xB)) +
        xB * xC * (L_BC * 0.5 * (1 + xC - xB) + L_CB * 0.5 * (1 + xB - xC)) * factor_BC +
        xC * xA * (L_CA * 0.5 * (1 + xA - xC) + L_AC * 0.5 * (1 + xC - xA)) * factor_AC +
        xA * xB * xC * L_ABC
    )

    return G_ideal + G_excess

#======================= Gibbs Wrappers ============================
def G_func(x, T, R, *L_and_P):
    return gibbs_ternary_subregular(x[0], x[1], x[2], T, R, *L_and_P)

def Gfunc(x):
    #print(x)
    x = np.stack(x)
    if len(x.shape)==2:
        if x.shape[0]==1:
            x = x[0]
    return gibbs_ternary_subregular(x[0], x[1], x[2], T, R, L0_AB, L1_AB, L0_BA, L1_BA, L_BC, L_CB, L_CA, L_AC, tau_BC, ppi_BC, L_ABC, PGPa)

#======================= Chemical Potential ============================

def chemical_potential_finite_diff(G_func, x, T, R, *L_and_P, h=1.0e-10):
    mu = np.zeros(3)
    for i in range(2):
        dx = np.zeros(3)
        # scale h to be ~fraction of the composition
        local_h = h * max(abs(x[i]), 1e-8)   #1e-8
        dx[i] = local_h
        dx[2] = -local_h
        G_plus = G_func(x + dx, T, R, *L_and_P)
        G_minus = G_func(x - dx, T, R, *L_and_P)
        mu[i] = (G_plus - G_minus) / (2 * local_h)
    mu[2] = -mu[0] - mu[1]
    return mu
    

#================================ CURVATURE CALCULATIONS =================================
# Calculate spinodal contour points where det(H) = 0
def calc_spinodal(T_val, P_val,L0_AB, L1_AB, L0_BA, L1_BA,L_BC, L_CB, L_CA, L_AC, tau_BC, ppi_BC, L_ABC, N):
    """
    Compute spinodal contour points (where det(H) = 0) in ternary space.
    Returns: array of [xA, xB, xC] points.
    """
    xA, xB, T, P = sp.symbols('xA xB T P', real=True, positive=True)
    xC = 1 - xA - xB
    R = 8.314
    factor_BC = (1 - T / tau_BC + P / ppi_BC)
    factor_AC = 1.0
    G_ideal = R * T * (xA*sp.log(xA) + xB*sp.log(xB) + xC*sp.log(xC))
    
    # Pressure enhancement of H2 in Fe
    L_AB = L0_AB + L1_AB * P
    L_BA = L0_BA + L1_BA * P
    
    G_excess = (
        xA * xB * (L_AB * 0.5*(1+xB-xA) + L_BA * 0.5*(1+xA-xB)) +
        xB * xC * (L_BC * 0.5*(1+xC-xB) + L_CB * 0.5*(1+xB-xC)) * factor_BC +
        xC * xA * (L_CA * 0.5*(1+xA-xC) + L_AC * 0.5*(1+xC-xA)) * factor_AC +
        xA * xB * xC * L_ABC
    )
    G_total = G_ideal + G_excess
    grad_xA = sp.diff(G_total, xA)
    grad_xB = sp.diff(G_total, xB)
    H11 = sp.diff(grad_xA, xA)
    H12 = sp.diff(grad_xA, xB)
    H21 = sp.diff(grad_xB, xA)
    H22 = sp.diff(grad_xB, xB)
    H = sp.Matrix([[H11, H12], [H21, H22]])
    det_H = H.det().simplify()
    det_H_func = sp.lambdify((xA, xB, T, P), det_H, modules='numpy')

    xA_vals = np.linspace(0, 1, N)
    xB_vals = np.linspace(0, 1, N)
    X, Y = np.meshgrid(xA_vals, xB_vals)
    Z = np.full_like(X, np.nan)

    for i in range(N):
        for j in range(N):
            xa = X[i, j]
            xb = Y[i, j]
            if xa + xb >= 1.0:
                continue
            try:
                Z[i, j] = det_H_func(xa, xb, T_val, P_val)
            except:
                Z[i, j] = np.nan

    plt.ioff()
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, levels=[0.00], colors='none')
    plt.close(fig)

    contour_segments = CS.allsegs[0]
    contour_points = []
    for segment in contour_segments:
        for xA_val, xB_val in segment:
            xC_val = 1.0 - xA_val - xB_val
            if xC_val >= 1e-6:
                contour_points.append([xA_val, xB_val, xC_val])

    return np.array(contour_points)


def evaluate_detH_grid(T_val, P_val, L0_AB, L1_AB, L0_BA, L1_BA, L_BC, L_CB, L_CA, L_AC, tau_BC, ppi_BC, L_ABC,  N):
    """
    Evaluates det(H) on a ternary grid and returns all points with det(H) values.
    """
    # Symbolic setup
    xA, xB, T, P = sp.symbols('xA xB T P', real=True, positive=True)
    xC = 1 - xA - xB
    R = 8.314

    factor_BC = (1 - T / tau_BC + P / ppi_BC)
    factor_AC = 1.0
    # Pressure enhancement of H2 in Fe
    L_AB = L0_AB + L1_AB * P
    L_BA = L0_BA + L1_BA * P

    G_ideal = R * T * (xA*sp.log(xA) + xB*sp.log(xB) + xC*sp.log(xC))
    G_excess = (
        xA * xB * (L_AB * 0.5*(1+xB-xA) + L_BA * 0.5*(1+xA-xB)) +
        xB * xC * (L_BC * 0.5*(1+xC-xB) + L_CB * 0.5*(1+xB-xC)) * factor_BC +
        xC * xA * (L_CA * 0.5*(1+xA-xC) + L_AC * 0.5*(1+xC-xA)) * factor_AC +
        xA * xB * xC * L_ABC
    )
    G_total = G_ideal + G_excess

    grad_xA = sp.diff(G_total, xA)
    grad_xB = sp.diff(G_total, xB)
    H = sp.Matrix([[sp.diff(grad_xA, xA), sp.diff(grad_xA, xB)],
                   [sp.diff(grad_xB, xA), sp.diff(grad_xB, xB)]])
    det_H_expr = H.det().simplify()
    det_H_func = sp.lambdify((xA, xB, T, P), det_H_expr, modules='numpy')

    # Evaluate on grid
    data = []
    for i in range(N + 1):
        for j in range(N + 1 - i):
            xa = i / N
            xb = j / N
            xc = 1.0 - xa - xb
            if xc < 0:
                continue
            try:
                if det_H_func(xa, xb, T_val, P_val) < 0:
                    data.append([xa, xb, xc])
            except:
                continue

    return np.array(data)


x_vals, y_vals, G_vals = [], [], []
points_ABC = []
xx = []
gr = []

for i in range(N+1):
    for j in range(N+1 - i):
        k = N - i - j
        xA = i / N
        xB = j / N
        xC = k / N
        xx.append([xA,xB,xC])
        gr.append([i,j])
        G = gibbs_ternary_subregular(xA, xB, xC, T, R, L0_AB, L1_AB, L0_BA, L1_BA, L_BC, L_CB, L_CA, L_AC, tau_BC, ppi_BC, L_ABC, PGPa)
        # Barycentric to 2D triangle
        x = 0.5 * (2 * xB + xC)
        y = (np.sqrt(3)/2) * xC
        x_vals.append(x)
        y_vals.append(y)
        G_vals.append(G)
        points_ABC.append((x, y, G))

# === Determine G offset ===
G_min = min(G_vals)
z_base = G_min - 1e4  # Move triangle 10 kJ/mol below min G

# Define a consistent offset below z_base
z_range = max(G_vals) - min(G_vals)
label_offset = 0.075 * z_range  # 5% of G range

z_label = z_base - label_offset

def wtpercent_to_mole_fractions(wt_MgSiO3, wt_Fe, wt_H2):
    # Molar masses in g/mol
    M_MgSiO3 = 100.36
    M_Fe = 55.845
    M_H2 = 2.016

    # Grams of each component in 100 g total
    grams_MgSiO3 = wt_MgSiO3
    grams_Fe = wt_Fe
    grams_H2 = wt_H2

    # Moles
    n_MgSiO3 = grams_MgSiO3 / M_MgSiO3
    n_Fe = grams_Fe / M_Fe
    n_H2 = grams_H2 / M_H2

    # Sum of moles
    n_total = n_MgSiO3 + n_Fe + n_H2
    if n_total <= 0:
        raise ValueError("Total moles computed as zero or negative; check input weights.")

    # Mole fractions
    x_MgSiO3 = n_MgSiO3 / n_total
    x_Fe = n_Fe / n_total
    x_H2 = n_H2 / n_total

    return x_MgSiO3, x_Fe, x_H2

def compute_subneptune_coords_(wtpercentH2):
    """
    Get (x_Fe, x_H2, x_silicate) mole fraction given input wt percent H2.
    Keeps MgSiO3:Fe at 2:1 by *mass* for the non-H2 remainder.
    """
    try:
        wt_H2 = wtpercentH2
        remaining = 100.0 - wt_H2
        if remaining < 0:
            remaining = 0.0
        wt_MgSiO3 = (0.667) * remaining
        wt_Fe = (0.333) * remaining

        x_MgSiO3, x_Fe, x_H2 = wtpercent_to_mole_fractions(wt_MgSiO3, wt_Fe, wt_H2)
        return (float(x_Fe), float(x_H2), float(x_MgSiO3))
    except Exception as e:
        print(f"[warn] Could not compute sub-Neptune coords from SUMMARY: {e}")
        # Fallback to previous defaults
        fe = 0.2581
        h2 = 0.453
        sil = 1.0 - fe - h2
        return (fe, h2, sil)

# =================================== Make 3D plot ======================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(x_vals, y_vals, G_vals,cmap='viridis', edgecolor='white',linewidth=0.05,alpha = 0.8)
# try cmap = 'viridis'

# === Draw triangle at z = z_base ===
triangle_base = np.array([
    [0.0, 0.0, z_base],            # A
    [1.0, 0.0, z_base],            # B
    [0.5, np.sqrt(3)/2, z_base],   # C
    [0.0, 0.0, z_base]             # back to A to close
])
ax.plot(triangle_base[:,0], triangle_base[:,1], triangle_base[:,2], 'k-', lw=2)

# === Label apices on base ===
ax.text(0.0, 0.0, z_label, "Fe$^{\circ}$", fontsize=14, ha='right', va='top', weight='bold')
ax.text(1.0, 0.0, z_label, "H$_2$", fontsize=14, ha='left', va='top', weight='bold')
ax.text(0.5, np.sqrt(3)/2, z_label , "MgSiO$_\mathbf{3}$", fontsize=14, ha='center', va='top', weight='bold')

# === Edge surfaces: A–B, B–C, C–A ===
def draw_edge_wall(xA_list, xB_list, xC_list, color):
    x2D = 0.5 * (2 * xB_list + xC_list)
    y2D = (np.sqrt(3)/2) * xC_list
    G3D = [gibbs_ternary_subregular(
        a, b, c, T, R, L0_AB, L1_AB, L0_BA, L1_BA, L_BC, L_CB, L_CA, L_AC, tau_BC, ppi_BC, L_ABC, PGPa
    ) for a, b, c in zip(xA_list, xB_list, xC_list)]

    verts = []
    for x, y, z in zip(x2D, y2D, G3D):
        verts.append((x, y, z))
    for x, y in zip(x2D[::-1], y2D[::-1]):
        verts.append((x, y, z_base))

    wall = Poly3DCollection([verts], alpha=0.6, facecolor='silver', linewidths=0.5)
    ax.add_collection3d(wall)

# A–B edge (C = 0)
xA = np.linspace(0, 1, 100)
xB = 1 - xA
xC = np.zeros_like(xA)
draw_edge_wall(xA, xB, xC, 'gray')

# B–C edge (A = 0)
xB = np.linspace(0, 1, 100)
xC = 1 - xB
xA = np.zeros_like(xB)
draw_edge_wall(xA, xB, xC, 'gray')

# C–A edge (B = 0)
xC = np.linspace(0, 1, 100)
xA = 1 - xC
xB = np.zeros_like(xC)
draw_edge_wall(xA, xB, xC, 'gray')

# === Final plot settings ===
ax.set_title(f'3D Gibbs Surface (T = {T:.0f} K, P = {PGPa:.1f} GPa)', fontsize=14)
#ax.set_xlabel('B–C Axis')
#ax.set_ylabel('C Apex Height')
#ax.set_zlabel('G$_{\mathtext{mix}}$ (J/mol)', labelpad = -50)
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad = 0.01)
cbar.set_label(r'G$_{\rm mix}$ (J/mol)', fontsize=14)
cbar.ax.tick_params(labelsize=12)  # optional: adjust tick label size too
plt.tight_layout()

# === Hide default cartesian ticks ===
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.xaxis.line.set_lw(0.)
ax.yaxis.line.set_lw(0.)
ax.zaxis.line.set_lw(0.)
ax.grid(False)

# === Remove Cartesian coordinate planes (make panes transparent) ===
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Optional: also remove the edges of those panes
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

# === Draw ternary-style ticks on triangle edges ===
tick_labels = np.linspace(0, 1, 6)
tick_fmt = lambda v: f"{v:.1f}"
tick_offset = 4000  # same as before

# A–B edge: horizontal → move downward
for i, val in enumerate(tick_labels[1:-1]):
    x = val
    y = 0
    ax.text(x, y, z_base - tick_offset, tick_fmt(1 - val),
            ha='center', va='top', fontsize=10)

# B–C edge: sloped up → nudge below and away from wall
for i, val in enumerate(tick_labels[1:-1]):
    xb = 1 - val
    xc = val
    x = 0.5 * (2 * xb + xc)
    y = (np.sqrt(3)/2) * xc
    ax.text(x, y, z_base - tick_offset * 0.5, tick_fmt(1 - xb),
            ha='left', va='top', fontsize=10, rotation=60)

# C–A edge: sloped up → push away at same z
for i, val in enumerate(tick_labels[1:-1]):
    xc = 1 - val
    xa = val
    x = 0.5 * xc
    y = (np.sqrt(3)/2) * xc
    ax.text(x, y, z_base - tick_offset * 0.5, tick_fmt(val),
            ha='right', va='top', fontsize=10, rotation=-60)

#=============================== Compute and plot spinodal points ===============================
N = 1000
stamp("starting spinodal")
spinodal_points = calc_spinodal(T, PGPa, L0_AB, L1_AB, L0_BA, L1_BA, L_BC, L_CB, L_CA, L_AC, tau_BC, ppi_BC, L_ABC,  N)

# Extract xA, xB, xC from the spinodal
A_vals_sp = spinodal_points[:, 0]
B_vals_sp = spinodal_points[:, 1]
C_vals_sp = spinodal_points[:, 2]

# Project ternary (xA, xB, xC) → 2D triangle coordinates
x_sp = 0.5 * (2 * B_vals_sp + C_vals_sp)
y_sp = (np.sqrt(3)/2) * C_vals_sp

# Compute Gibbs energy at each spinodal point
G_sp = [gibbs_ternary_subregular(a, b, c, T, R, L0_AB, L1_AB, L0_BA, L1_BA, L_BC, L_CB, L_CA, L_AC, tau_BC, ppi_BC, L_ABC, PGPa)
        for a, b, c in zip(A_vals_sp, B_vals_sp, C_vals_sp)]

stamp("spinodal done")

# --- Plot the spinodal points on top of the 3D Gibbs surface ---
ax.scatter(x_sp, y_sp, np.array(G_sp)+0.05*abs(max(G_sp)), color='white', s=2, label='Spinodal Points', depthshade=False)


# ================================= Save 3D surface plot ====================================
suffix = '_ternary_solvus_models'
folder_name = suffix.strip("_")
os.makedirs(folder_name, exist_ok=True)
plotname = f"3D_ternary_T{int(T)}K_P{PGPa:.1f}GPa.pdf"
# Generate the new file path by placing the file in the suffix-named folder
new_file_path = os.path.join(folder_name, plotname)
#plt.savefig(new_file_path,bbox_inches='tight',dpi=1000)

# Save spinodal points to file
# Combine the data into columns
data = np.column_stack((A_vals_sp, B_vals_sp, C_vals_sp, G_sp))

# Define header and file path
header = "xA    xB    xC    G_mix (J/mol)"
suffix = '_ternary_solvus_models'
folder_name = suffix.strip("_")
filename = f"spinodal_G_values_T{int(T)}K_P{PGPa:.1f}GPa.txt"
output_path = os.path.join(folder_name, filename)

# Write to file
#np.savetxt(output_path, data, header=header, fmt="%.6f", delimiter="\t")

print(f"Spinodal Gibbs values saved to {output_path}")



#================================== Ternary plot ====================================
points = list(zip(A_vals_sp, B_vals_sp, C_vals_sp))

detH_data = evaluate_detH_grid(T, PGPa, L0_AB, L1_AB, L0_BA, L1_BA, L_BC, L_CB, L_CA, L_AC, tau_BC, ppi_BC, L_ABC,  N=1000)

# === Plot setup
scale = 1.0  # normalized mole fractions
fig, tax = ternary.figure(scale=1.0)
#fig.set_size_inches(6, 5)  # Aspect ratio slightly taller for label spacing
tax.get_axes().set_aspect('equal')

# === Style
tax.boundary(linewidth=1.5)
#tax.gridlines(multiple=0.1, color="lightgray", linewidth=0.6)

# === Labels

fontsize = 14
offset = 0.05
tax.annotate("  Fe", position=(1+offset, 0, 0), fontsize=fontsize, ha='left', va='center')
tax.annotate("            H₂", position=(0, 1+0.01, 0), fontsize=fontsize, ha='center', va='bottom')
tax.annotate("MgSiO$_{3}$                    ", position=(0, 0, 1), fontsize=fontsize, ha='center', va='top')


# --- Add shaded convex region (det(H) < 0) as light grey ---
tax.scatter(detH_data, marker='o', color='lightgrey', s=2, edgecolors='lightgrey', linewidths=0.3)
reset_stamp()                      # zero the timer whenever you like
stamp("starting spinodal plot on ternary")
# === Data points for spinodal
tax.scatter(points, marker='o', color='red', s=2, edgecolors='k', linewidths=0.3)
stamp("spinodal plot done")


# Add planet markers
# Example coordinates — update with real compositions
earth_coords = (0.444, 0.0593, (1-0.444-0.0593))    # (Fe, H2, silicate)
#subneptune_coords = (0.2581, 0.47, (1-0.2581-0.47)) # ~2 wt % H2
#subneptune_coords_circ = (0.2581, 0.47, (1-0.2581-0.47)) # ~2 wt % H2

xFe,xH2,xMgSiO3 = compute_subneptune_coords_(wtpercent_H2)
subneptune_coords = (xFe, xH2, xMgSiO3) # set at input at top
subneptune_coords_circ = (xFe, xH2, xMgSiO3) #set by input at top
neptune_coords = (0.0437, 0.9077, (1-0.0437-0.9077))

# Earth: 🜨 (U+1F728) — use text if Unicode symbol not supported
tax.annotate(r"$\oplus$", position=earth_coords, fontsize=18, ha='center', va='center', color='black')
tax.annotate("♆", position=subneptune_coords, fontsize=18, ha='center', va='center', color='green')
tax.annotate("♆", position=neptune_coords, fontsize=18, ha='center', va='center', color='green')
# Plot an open circle around the symbol
tax.scatter([subneptune_coords_circ], marker='o', facecolors='none', edgecolors='green', s=190, linewidths=1.5)
#tax.annotate(r"$\mathrm{Sub}\text{-}\!\!{\large \neptune}$", position=neptune_coords, fontsize=18, ha='center', va='center', color='green')



#=========================== Find binodes using convex hull ================================
from phasehull import *  # Kees' phasehull python script

#niter    = 3   # Nr of refinement iterations / levels
niter    = 3
nfact    = 2   # Refinement factor, 2
nspan    = 2   # Span of refinement (how many course grid points to left and right to refine), 2

reset_stamp
stamp("Begin points for binodals")

xx       = np.stack(xx)
nres0    = int(np.round(1/np.abs(xx[1]-xx[0]).max()))
Solution = Liquid("Solution",["Fe","H2","MgSiO3"],Gfunc)
phull    = PhaseHull(["Fe","H2","MgSiO3"],liquids=Solution,xgrid=xx,nres0=nres0,nrefine=niter,nfact=nfact,nspan=nspan)

stamp("End binodal point calculation")

print('Done with the calculations. Now doing the plotting.')


# ───────────────── BINODAL OUTLINE (stype boundary) ─────────────────

binodals = phull.select_simplices_of_a_given_kind('tieline_c0l3')
phull.compute_tie_lines_from_simplices(binodals)
groups   = phull.sort_tie_lines(binodals,'tieline_c0l3')

# ---------------- Plot binodal and tie lines ----------------------
# functions for fill

# ----- Utility: make sure every point is a ternary triple (a,b,c) -----------
def _as_triple(p):
    p = np.asarray(p, dtype=float).ravel()
    if p.size == 3:                # already (a,b,c)
        return tuple(p)
    if p.size == 2:                # only (xA,xB) given → add xC
        return (p[0], p[1], 1.0 - p[0] - p[1])
    raise ValueError("Points must have 2 or 3 coordinates.")

# ---------- New fill function that plays nicely with python-ternary -------------
def ternary_fill(x, stype, tax, scale=1,
                 done=False, color=None, lincol=None, **kw):
    """
    x      : (N,2) or (N,3) array of polygon vertices (clockwise / ccw)
    stype  : key used for colour maps / legend label
    tax    : python-ternary TernaryAxesSubplot  (i.e. the 'tax' object)
    scale  : plot scale (same one you passed to ternary.figure)
    done   : if True, skip the legend entry
    color  : face colour  (falls back on colour map below)
    lincol : outline colour  (optional)
    **kw   : forwarded to ax.fill (alpha, zorder, etc.)
    """
    # ---------- colour lookup tables (keep your old choices) ----------
    face_colours = {
        'allcryst'       : 'C1', 'liquid'        : 'C0',
        'cryst_1_liq_1'  : 'C4', 'cryst_1_liq_2' : 'C9',
        'cryst_2_liq_1'  : 'C6', 'crystals'      : 'C3',
        'inmisc_liquids' : 'C9'
    }
    line_colours = {
        'allcryst'       : 'C5',
        'cryst_1_liq_2'  : 'C0'
    }

    if color  is None:  color  = face_colours.get(stype, 'C7')
    if lincol is None:  lincol = line_colours.get(stype, None)

    # ---------- convert to Cartesian once --------------------------------
    triples = np.array([_as_triple(p) for p in x])
    a, b, c = triples.T
    x_cart  = (a + 0.5*b) * scale
    y_cart  = (np.sqrt(3)/2 * b) * scale

    # close the polygon
    x_cart = np.append(x_cart, x_cart[0])
    y_cart = np.append(y_cart, y_cart[0])

    # ---------- draw on the *underlying* Matplotlib Axes -----------------
    ax = tax.get_axes()            # plain Axes tied to the ternary subplot
    label = None if done else stype
    ax.fill(x_cart, y_cart, color=color, label=label, **kw)
    if lincol is not None:
        ax.plot(x_cart, y_cart, color=lincol, linewidth=0.5)

    return ax      # (optional) handy if you want to tweak it later



# ----- New versions of plotting functions that accept `tax` --------------
def _as_triple(p):
    """
    Accept (xA,xB)  or (xA,xB,xC) and return a 3-tuple (a,b,c).
    """
    p = np.asarray(p, dtype=float).ravel()
    if p.size == 3:
        return tuple(p)                       # already a triple
    elif p.size == 2:
        return (p[0], p[1], 1.0 - p[0] - p[1])
    else:
        raise ValueError("Each point must have 2 or 3 coordinates.")

#def plot_tie_lines(xx, tax, **kw):
#    for p, q in xx:                           # shape (N,2,*)  – two endpoints
#        tax.line(_as_triple(p), _as_triple(q), **kw)
        
def plot_tie_lines(xx, tax, **kw):
    """
    xx shape: (N, 2, *)  – each row has two endpoints (xA,xB[,xC])
    Draws the line and puts a marker at each end.
    Extra keyword args in **kw are passed to both line and scatter,
    but you can override marker properties inside the function.
    """
    # line style (use whatever came in via **kw)
    line_kw = kw.copy()

    # marker style – inherit colour but tweak size/shape
    marker_kw = kw.copy()
    marker_kw.setdefault("marker", "o")
    marker_kw.setdefault("s",      20)      # scatter size
    marker_kw.setdefault("edgecolors", "none")

    for p, q in xx:
        p3, q3 = _as_triple(p), _as_triple(q)

        # --- segment ---
        tax.line(p3, q3, **line_kw)

        # --- endpoints ---
        # scatter expects a 2-D list of points, so wrap each triple in [...]
        tax.scatter([p3], **marker_kw)
        tax.scatter([q3], **marker_kw)
        
        

def plot_binodal_curve(xx, tax, **kw):
    left   = xx[:, 0]                         # (N,*) left ends
    right  = xx[::-1, 1]                      # (N,*) right ends (reversed)
    pts    = np.vstack([left, right, left[:1]])
    trip   = [_as_triple(row) for row in pts]
    tax.plot(trip, **kw)


# PLOT BINODAL AND BINODES WITH TIE LINES

#stride        = 500 # 64, 800, inteval as in "every strideth" line
stride        = 50 # 64, 800, inteval as in "every strideth" line
color_curve   = "darkblue"
color_tieline = "darkblue"

for g in groups:                                      # each g is one binodal group
    if len(g) > 10:                                    # need ≥2 tie-lines to trace a curve
        # -- 1. smooth outline of the binodal ---------------------------------
        xx_all = phull.get_tie_lines_x_values_for_a_group(
                 g, stride=2)                    # keeps every point if stride = 1
        plot_binodal_curve(xx_all,tax, color=color_curve,   lw=1.0)

        # -- 2. sparsified tie-lines for clarity ------------------------------
        xx_sparse = phull.get_tie_lines_x_values_for_a_group(
                    g, stride=stride)
        plot_tie_lines   (xx_sparse, tax, color=color_tieline, lw=1.0)
        
        # ---- fill the enclosed two-phase region ----------------------
        left   = xx_all[:, 0, :]
        right  = xx_all[::-1, 1, :]
        poly   = np.vstack([left, right])          # (2N,2) polygon
        ternary_fill(poly, 'inmisc_liquids', tax,
                     color='skyblue', alpha=0.65, zorder=-5, done=True)

#============================= End convex hull calculations ============================

# === Final polish on ternary plot
tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')  # No bounding box
tax._redraw_labels()

# === Add T and P label (adjust x, y to position the text)
text_label = f"T = {T:.0f} K, P = {PGPa:.1f} GPa"
fig.text(0.05, 0.92, text_label, fontsize=12, ha='left', va='top')

# === Tick marks
tax.ticks(
    multiple=0.1,                # Tick interval (e.g., every 0.1 = 10%)
    axis='lbr',                  # Show ticks on all three axes: left, bottom, right
    linewidth=1,
    clockwise=True,              # Match the orientation of the diagram
    tick_formats="%.1f",         # Tick label format (1 decimal place)
    fontsize=10,
    offset=0.02                 # Small offset so labels don't collide with axes
# Length of the tick marks in points
)

plt.tight_layout(pad=1.0)

suffix = '_ternary_solvus_models'
folder_name = suffix.strip("_")
os.makedirs(folder_name, exist_ok=True)
plotname = f"binodal_ternary_T{int(T)}K_P{PGPa:.1f}GPa.png"
# Generate the new file path by placing the file in the suffix-named folder
new_file_path = os.path.join(folder_name, plotname)
plt.savefig(new_file_path,bbox_inches='tight',dpi=350)



print(f"Ternary projection saved to {new_file_path}")

#======================= Show final ternary plot ================================
plt.show()


