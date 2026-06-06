"""
Assignment 5 - Linear Algebra and Least Squares Regression
Matemáticas Avanzadas - MING, Universidad Tecnológica de Bolívar
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})

# =============================================================================
# PROBLEM 1: Recursive Determinant via Cofactor Expansion
# =============================================================================

def my_rec_det(M):
    """
    Computes det(M) recursively using cofactor expansion along the first row.
    det(M) = sum_{i=1}^{n} (-1)^{i-1} * M[0,i-1] * det(minor)
    """
    M = np.array(M, dtype=float)
    n = M.shape[0]
    # Base cases
    if n == 1:
        return M[0, 0]
    if n == 2:
        return M[0, 0]*M[1, 1] - M[0, 1]*M[1, 0]
    # Recursive cofactor expansion along first row
    det_val = 0.0
    for i in range(n):
        # Minor: remove row 0 and column i
        minor = np.delete(np.delete(M, 0, axis=0), i, axis=1)
        cofactor = ((-1) ** i) * my_rec_det(minor)
        det_val += M[0, i] * cofactor
    return det_val

# --- Validation ---
print("=" * 60)
print("PROBLEM 1: Recursive Determinant")
print("=" * 60)

M2 = np.array([[3, 1],
               [2, 4]])
M3 = np.array([[1, 2, 3],
               [0, 4, 5],
               [1, 0, 6]])
M4 = np.array([[0, 2, 1, 3],
               [3, 2, 8, 1],
               [1, 0, 0, 3],
               [0, 3, 2, 1]])

for M, name in [(M2, "2x2"), (M3, "3x3"), (M4, "4x4 (Kong ex.)")]:
    d_rec = my_rec_det(M)
    d_np  = np.linalg.det(M)
    print(f"  {name}: my_rec_det = {d_rec:.4f} | np.linalg.det = {d_np:.4f} | match = {np.isclose(d_rec, d_np)}")

# --- Figure P1: verification table ---
fig1, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')

formula_str = r"$\det(M) = \sum_{i=1}^{n}(-1)^{i-1}\,M(1,i)\,\det(m_{1,i})$"

ax.text(0.5, 0.88, "Problem 1 – Recursive Determinant via Cofactor Expansion",
        fontsize=13, ha='center', va='center', fontweight='bold')
ax.text(0.5, 0.72, formula_str, fontsize=13, ha='center', va='center')

# Table of results
cols = ['Matrix', 'my_rec_det', 'np.linalg.det', 'Match']
rows = [
    ['2x2', '10.0', '10.0', 'True'],
    ['3x3', '22.0', '22.0', 'True'],
    ['4x4 (Kong)', '-38.0', '-38.0', 'True'],
]
table = ax.table(cellText=rows, colLabels=cols,
                 loc='lower center', cellLoc='center',
                 bbox=[0.05, 0.05, 0.90, 0.52])
table.auto_set_font_size(False)
table.set_fontsize(11)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold')
    elif rows[r-1][3] == 'True':
        cell.set_facecolor('#E8F5E9')

fig1.tight_layout()
fig1.savefig("fig_p1_det.png", bbox_inches='tight')
plt.close(fig1)
print("  Figure P1 saved.\n")


# =============================================================================
# PROBLEM 2: Network Flow Calculator
# =============================================================================
#
# Graph topology (confirmed):
#   f1: S1 → N4    f2: N4 → N3    f3: N1 → N4
#   f4: S2 → N1    f5: S2 → N2    f6: N1 → N5    f7: N2 → N5
#
# Conservation equations (inflow = outflow + demand):
#   S1:  f1            = S[0]
#   S2:  f4 + f5       = S[1]
#   N1:  f4            = f3 + f6 + d[0]
#   N2:  f5            = f7 + d[1]
#   N3:  f2            = d[2]           (f2 determined directly)
#   N4:  f1 + f3       = f2 + d[3]
#   N5:  f6 + f7       = d[4]
#
# Unknowns: f = [f1, f2, f3, f4, f5, f6, f7]
# System: 7 eq, 7 unknowns — rank 6 (ΣS = Σd guarantees redundancy)
# Strategy: use np.linalg.lstsq (minimum-norm solution)

def my_flow_calculator(S, d):
    """
    Solves the power flow network.
    S : array-like, shape (2,)  — supply station capacities [S1, S2]
    d : array-like, shape (5,)  — node demands [d1..d5]
    Returns f : array, shape (7,) — flows [f1..f7]
    """
    S = np.array(S, dtype=float).flatten()
    d = np.array(d, dtype=float).flatten()

    # Coefficient matrix A (rows = equations, cols = [f1..f7])
    #        f1  f2  f3  f4  f5  f6  f7
    A = np.array([
        [ 1,  0,  0,  0,  0,  0,  0],   # S1:  f1 = S[0]
        [ 0,  0,  0,  1,  1,  0,  0],   # S2:  f4+f5 = S[1]
        [ 0,  0, -1,  1,  0, -1,  0],   # N1:  f4 = f3+f6+d[0]  → f4-f3-f6 = d[0]
        [ 0,  0,  0,  0,  1,  0, -1],   # N2:  f5 = f7+d[1]     → f5-f7 = d[1]
        [ 0,  1,  0,  0,  0,  0,  0],   # N3:  f2 = d[2]
        [ 1,  -1, 1,  0,  0,  0,  0],   # N4:  f1+f3 = f2+d[3]  → f1+f3-f2 = d[3]
        [ 0,  0,  0,  0,  0,  1,  1],   # N5:  f6+f7 = d[4]
    ], dtype=float)

    b = np.array([S[0], S[1], d[0], d[1], d[2], d[3], d[4]], dtype=float)

    # Minimum-norm least squares solution
    f, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return f

print("=" * 60)
print("PROBLEM 2: Network Flow Calculator")
print("=" * 60)

S_test1 = np.array([10, 10])
d_test1 = np.array([4, 4, 4, 4, 4])
f1_out = my_flow_calculator(S_test1, d_test1)
print(f"  Test 1 — S={S_test1}, d={d_test1}")
print(f"  f = {np.round(f1_out, 4)}")

S_test2 = np.array([10, 10])
d_test2 = np.array([3, 4, 5, 4, 4])
f2_out = my_flow_calculator(S_test2, d_test2)
print(f"  Test 2 — S={S_test2}, d={d_test2}")
print(f"  f = {np.round(f2_out, 4)}")

# Verify conservation for test 1
f = f1_out
print(f"\n  Verification Test 1:")
print(f"    S1 check: f1 = {f[0]:.2f} (expected {S_test1[0]})")
print(f"    S2 check: f4+f5 = {f[3]+f[4]:.2f} (expected {S_test1[1]})")
print(f"    N1 check: f4-f3-f6 = {f[3]-f[2]-f[5]:.2f} (expected {d_test1[0]})")
print(f"    N2 check: f5-f7 = {f[4]-f[6]:.2f} (expected {d_test1[1]})")
print(f"    N3 check: f2 = {f[1]:.2f} (expected {d_test1[2]})")
print(f"    N4 check: f1+f3-f2 = {f[0]+f[2]-f[1]:.2f} (expected {d_test1[3]})")
print(f"    N5 check: f6+f7 = {f[5]+f[6]:.2f} (expected {d_test1[4]})")

# --- Figure P2: Network diagram ---
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.set_xlim(0, 10); ax2.set_ylim(0, 8); ax2.axis('off')
ax2.set_title("Problem 2 – Power Flow Network", fontweight='bold')

nodes = {
    'S1': (1.5, 6.5), 'S2': (8.5, 6.5),
    'N1': (5.0, 6.0), 'N2': (7.5, 4.5),
    'N3': (2.0, 1.5), 'N4': (4.0, 3.5),
    'N5': (6.0, 1.5),
}
colors = {'S1': '#4CAF50', 'S2': '#4CAF50',
          'N1': '#2196F3', 'N2': '#2196F3',
          'N3': '#2196F3', 'N4': '#2196F3', 'N5': '#2196F3'}

for name, (x, y) in nodes.items():
    circle = plt.Circle((x, y), 0.45, color=colors[name], zorder=3)
    ax2.add_patch(circle)
    ax2.text(x, y, name, ha='center', va='center',
             color='white', fontweight='bold', fontsize=10, zorder=4)

edges = [
    ('S1', 'N4', 'f₁'), ('N4', 'N3', 'f₂'), ('N1', 'N4', 'f₃'),
    ('S2', 'N1', 'f₄'), ('S2', 'N2', 'f₅'), ('N1', 'N5', 'f₆'),
    ('N2', 'N5', 'f₇'),
]
for src, dst, label in edges:
    x1, y1 = nodes[src]; x2, y2 = nodes[dst]
    dx, dy = x2-x1, y2-y1
    ax2.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle='->', color='#333', lw=1.8),
                 zorder=2)
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax2.text(mx+0.15, my+0.15, label, fontsize=9,
             color='#C62828', fontweight='bold')

f_vals = np.round(f1_out, 2)
flow_labels = [f'f₁={f_vals[0]}', f'f₂={f_vals[1]}', f'f₃={f_vals[2]}',
               f'f₄={f_vals[3]}', f'f₅={f_vals[4]}', f'f₆={f_vals[5]}', f'f₇={f_vals[6]}']
ax2.text(0.5, 0.3, "Test 1 flows: " + ", ".join(flow_labels),
         transform=ax2.transAxes, fontsize=8.5, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig2.tight_layout()
fig2.savefig("fig_p2_network.png", bbox_inches='tight')
plt.close(fig2)
print("\n  Figure P2 saved.\n")


# =============================================================================
# PROBLEM 3: Gauss Elimination with Partial Pivoting
# =============================================================================

def gauss_elimination_pivot(A, b):
    """
    Solves Ax = b using Gaussian elimination with partial pivoting.
    Returns x (solution), steps (augmented matrix trace), and det_A.
    """
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])
    sign = 1  # track sign for determinant

    for col in range(n):
        # Partial pivoting: find row with max absolute value in current column
        max_row = col + np.argmax(np.abs(Ab[col:, col]))
        if max_row != col:
            Ab[[col, max_row]] = Ab[[max_row, col]]
            sign *= -1

        if np.abs(Ab[col, col]) < 1e-12:
            raise ValueError(f"Matrix is singular or nearly singular at column {col}.")

        # Eliminate below pivot
        for row in range(col + 1, n):
            factor = Ab[row, col] / Ab[col, col]
            Ab[row, col:] -= factor * Ab[col, col:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]

    # Determinant = sign * product of diagonal of U
    det_A = sign * np.prod(np.diag(Ab[:, :n]))
    return x, Ab, det_A

print("=" * 60)
print("PROBLEM 3: Gauss Elimination with Partial Pivoting")
print("=" * 60)

# --- System (a): homogeneous system ---
A3a = np.array([
    [0, 3, 5],
    [3, -4, 0],
    [5, 0, 6]
], dtype=float)
b3a = np.array([0, 0, 0], dtype=float)
print("\n  System (a):")
print("    0x1 + 3x2 + 5x3 = 0")
print("    3x1 - 4x2 + 0x3 = 0")
print("    5x1 + 0x2 + 6x3 = 0")
try:
    x3a, _, det3a = gauss_elimination_pivot(A3a, b3a)
    print(f"  Solution: x = {np.round(x3a, 6)}")
    print(f"  det(A) = {det3a:.4f}")
    print(f"  (Homogeneous system → trivial solution x=0 since det≠0)")
except Exception as e:
    print(f"  Error: {e}")

# --- System (b): non-homogeneous 4x4 ---
A3b = np.array([
    [6.4,  3.2,  0.0,  0.0],
    [3.2, -1.6,  4.8,  0.0],
    [0.0,  4.8, -9.6,  7.2],
    [0.0,  0.0,  7.2,  4.8]
], dtype=float)
b3b = np.array([-1.6, 32.0, -78.0, 20.4], dtype=float)
print("\n  System (b):")
print("    6.4x1 + 3.2x2              = -1.6")
print("    3.2x1 - 1.6x2 + 4.8x3     = 32.0")
print("          + 4.8x2 - 9.6x3 + 7.2x4 = -78.0")
print("                    7.2x3 + 4.8x4 = 20.4")

x3b, Ab3b, det3b = gauss_elimination_pivot(A3b, b3b)
print(f"  Solution: x = {np.round(x3b, 4)}")
print(f"  Verification A*x: {np.round(A3b @ x3b, 4)}")
print(f"  det(A) = {det3b:.4f}")

# --- Experiment: nearly-singular and large/sparse systems ---
print("\n  Experiment: nearly-singular matrix")
eps = 1e-10
A_near_sing = np.array([[1, 2], [1+eps, 2+eps]])
b_near = np.array([3, 3+eps], dtype=float)
try:
    x_ns, _, det_ns = gauss_elimination_pivot(A_near_sing, b_near)
    print(f"    det = {det_ns:.2e} → near-singular, solution may be unreliable: x = {np.round(x_ns, 4)}")
except Exception as e:
    print(f"    Singular detected: {e}")

print("\n  Experiment: 20x20 random dense system")
np.random.seed(42)
n_large = 20
A_large = np.random.randn(n_large, n_large)
x_true  = np.random.randn(n_large)
b_large = A_large @ x_true
x_large, _, _ = gauss_elimination_pivot(A_large, b_large)
err_large = np.linalg.norm(x_large - x_true)
print(f"    L2 error: {err_large:.2e}")

print("\n  Experiment: 50x50 sparse system (tridiagonal)")
n_sp = 50
A_sparse = (np.diag(4*np.ones(n_sp))
            + np.diag(-np.ones(n_sp-1), 1)
            + np.diag(-np.ones(n_sp-1), -1))
x_sp_true = np.ones(n_sp)
b_sp = A_sparse @ x_sp_true
x_sp, _, _ = gauss_elimination_pivot(A_sparse, b_sp)
err_sp = np.linalg.norm(x_sp - x_sp_true)
print(f"    L2 error (sparse 50x50): {err_sp:.2e}")

# --- Figures P3 ---
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: residuals for system (b)
residuals_b = A3b @ x3b - b3b
ax = axes3[0]
ax.bar([f'$r_{i+1}$' for i in range(4)], residuals_b, color='steelblue', edgecolor='black')
ax.axhline(0, color='red', linewidth=0.8, linestyle='--')
ax.set_title("P3(b) – Residuals after Gauss Elimination")
ax.set_ylabel("Residual value")
ax.set_xlabel("Equation index")

# Right: error vs system size
sizes = list(range(5, 101, 5))
errors = []
np.random.seed(0)
for n in sizes:
    A_ = np.random.randn(n, n) + n*np.eye(n)  # diagonally dominant
    x_ = np.random.randn(n)
    b_ = A_ @ x_
    xsol, _, _ = gauss_elimination_pivot(A_, b_)
    errors.append(np.linalg.norm(xsol - x_))

ax2_ = axes3[1]
ax2_.semilogy(sizes, errors, 'o-', color='darkorange', markersize=4)
ax2_.set_title("P3 – Solution error vs. system size")
ax2_.set_xlabel("System size n")
ax2_.set_ylabel("$\\|x_{sol} - x_{true}\\|_2$ (log scale)")
ax2_.grid(True, which='both', alpha=0.3)

fig3.tight_layout()
fig3.savefig("fig_p3_gauss.png", bbox_inches='tight')
plt.close(fig3)
print("\n  Figure P3 saved.\n")


# =============================================================================
# PROBLEM 4: Gauss-Seidel Iteration (3 steps)
# =============================================================================
#
# System: 10x1 + x2 + x3 = 6
#          x1 + 10x2 + x3 = 6
#          x1 +  x2 + 10x3 = 6
#
# Explicit form:
#   x1 = (6 - x2 - x3) / 10
#   x2 = (6 - x1 - x3) / 10
#   x3 = (6 - x1 - x2) / 10
# Exact solution: x1 = x2 = x3 = 0.5

def gauss_seidel_steps(A, b, x0, n_steps):
    """Performs exactly n_steps Gauss-Seidel iterations. Returns history."""
    n = len(b)
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    for _ in range(n_steps):
        for i in range(n):
            s = b[i] - sum(A[i, j]*x[j] for j in range(n) if j != i)
            x[i] = s / A[i, i]
        history.append(x.copy())
    return history

print("=" * 60)
print("PROBLEM 4: Gauss-Seidel (3 iterations)")
print("=" * 60)
print("  Exact solution: x1 = x2 = x3 = 6/12 = 0.5")

A4 = np.array([[10, 1, 1],
               [1, 10, 1],
               [1,  1, 10]], dtype=float)
b4 = np.array([6, 6, 6], dtype=float)

for label, x0 in [("a) x0 = [0,0,0]", [0,0,0]),
                   ("b) x0 = [10,10,10]", [10,10,10])]:
    hist = gauss_seidel_steps(A4, b4, x0, 3)
    print(f"\n  {label}")
    print(f"  {'k':>3} | {'x1':>10} {'x2':>10} {'x3':>10} | {'||e||2':>10}")
    exact = np.array([0.5, 0.5, 0.5])
    for k, xk in enumerate(hist):
        err = np.linalg.norm(xk - exact)
        print(f"  {k:>3} | {xk[0]:>10.6f} {xk[1]:>10.6f} {xk[2]:>10.6f} | {err:>10.6f}")

# --- Figure P4: convergence ---
# Run more steps for plot
hist_a = gauss_seidel_steps(A4, b4, [0,0,0], 15)
hist_b = gauss_seidel_steps(A4, b4, [10,10,10], 15)
exact4 = np.array([0.5, 0.5, 0.5])
err_a = [np.linalg.norm(x - exact4) for x in hist_a]
err_b = [np.linalg.norm(x - exact4) for x in hist_b]

fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: trajectories of x1 for both starts
ax4a = axes4[0]
for comp, col in enumerate(['#E53935', '#1E88E5', '#43A047']):
    ax4a.plot([h[comp] for h in hist_a], 'o-', color=col,
              label=f'$x_{comp+1}$ (start 0)', markersize=4)
    ax4a.plot([h[comp] for h in hist_b], 's--', color=col,
              label=f'$x_{comp+1}$ (start 10)', markersize=4, alpha=0.6)
ax4a.axhline(0.5, color='black', linestyle=':', lw=1.5, label='exact = 0.5')
ax4a.set_title("P4 – Gauss-Seidel: variable evolution")
ax4a.set_xlabel("Iteration k"); ax4a.set_ylabel("Value")
ax4a.legend(fontsize=7, ncol=2); ax4a.grid(alpha=0.3)

# Right: error norm
ax4b = axes4[1]
ax4b.semilogy(err_a, 'b-o', markersize=5, label='Start: [0,0,0]')
ax4b.semilogy(err_b, 'r-s', markersize=5, label='Start: [10,10,10]')
ax4b.axvline(3, color='gray', linestyle='--', lw=1, label='k=3')
ax4b.set_title("P4 – Gauss-Seidel: convergence ($\\|e\\|_2$)")
ax4b.set_xlabel("Iteration k"); ax4b.set_ylabel("$\\|x^{(k)} - x^*\\|_2$")
ax4b.legend(); ax4b.grid(True, which='both', alpha=0.3)

fig4.tight_layout()
fig4.savefig("fig_p4_seidel.png", bbox_inches='tight')
plt.close(fig4)
print("\n  Figure P4 saved.\n")


# =============================================================================
# PROBLEMS 5, 6, 7: Simple Linear Least Squares  β = (AᵀA)⁻¹AᵀY
# =============================================================================

def least_squares_1param(x_data, y_data, model='y=ax'):
    """
    Fits y = a*x  (model='y=ax') or y = a*x + b (model='y=ax+b')
    using β = (AᵀA)⁻¹AᵀY.
    Returns parameters and fitted values.
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    if model == 'y=ax':
        A = x.reshape(-1, 1)
    else:
        A = np.column_stack([x, np.ones_like(x)])
    beta = np.linalg.inv(A.T @ A) @ A.T @ y
    y_fit = A @ beta
    residuals = y - y_fit
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((y - np.mean(y))**2)
    R2 = 1 - SS_res/SS_tot
    return beta, y_fit, R2

print("=" * 60)
print("PROBLEM 5: Resistance R from U = R·i")
print("=" * 60)
i_data = np.array([2.0, 4.0, 6.0, 10.0])
U_data = np.array([104, 206, 314, 530], dtype=float)
beta5, Ufit5, R2_5 = least_squares_1param(i_data, U_data, 'y=ax')
R_est = beta5[0]
print(f"  Model: U = R·i")
print(f"  β = (AᵀA)⁻¹AᵀU → R ≈ {R_est:.4f} Ω")
print(f"  R² = {R2_5:.6f}")

# Show the normal equation explicitly
A5 = i_data.reshape(-1,1)
AtA5 = A5.T @ A5
AtY5 = A5.T @ U_data
print(f"  AᵀA = {AtA5[0,0]:.1f},  AᵀY = {AtY5[0]:.1f}")
print(f"  R = AᵀY / AᵀA = {AtY5[0]/AtA5[0,0]:.4f}")

print("\n" + "=" * 60)
print("PROBLEM 6: Average speed vav from s = vav·t")
print("=" * 60)
t_data = np.array([9.0, 10.0, 11.0, 12.0])
s_data = np.array([140.0, 220.0, 310.0, 410.0])

# Model 1: through origin  s = vav*t
beta6_0, sfit6_0, R2_6_0 = least_squares_1param(t_data, s_data, 'y=ax')
vav_0 = beta6_0[0]
print(f"  Model 1 (through origin): s = vav*t")
print(f"  vav = {vav_0:.4f} km/h  |  R2 = {R2_6_0:.6f}")
print(f"  Note: low R2 because data does not pass through origin.")

# Model 2: with intercept  s = vav*t + s0  (better fit)
beta6, sfit6, R2_6 = least_squares_1param(t_data, s_data, 'y=ax+b')
vav = beta6[0]
s0  = beta6[1]
print(f"\n  Model 2 (with intercept): s = vav*t + s0")
print(f"  vav = {vav:.4f} km/h,  s0 = {s0:.4f} km  |  R2 = {R2_6:.6f}")
print(f"  Kreyszig intended answer uses model with intercept.")

print("\n" + "=" * 60)
print("PROBLEM 7: Spring modulus k from F = k·s")
print("=" * 60)
s_spring = np.array([0.50, 1.02, 1.99, 3.01, 4.98, 10.03])
F_spring = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 20.0])
beta7, Ffit7, R2_7 = least_squares_1param(s_spring, F_spring, 'y=ax')
k_est = beta7[0]
print(f"  Model: F = k·s")
print(f"  k ≈ {k_est:.4f} lbf/cm")
print(f"  R² = {R2_7:.6f}")

# --- Figures P5, P6, P7 ---
fig567, axes567 = plt.subplots(1, 3, figsize=(14, 4.5))

for ax, xd, yd, yfit, param, xlabel, ylabel, title, unit in [
    (axes567[0], i_data, U_data, Ufit5,
     f'R = {R_est:.2f} Ω', 'Current $i$ [A]', 'Voltage $U$ [V]',
     'P5 – Resistance (U = R·i)', 'Ω'),
    (axes567[1], t_data, s_data, sfit6,
     f'$v_{{av}}$={vav:.1f} km/h, $s_0$={s0:.1f}', 'Time $t$ [h]', 'Distance $s$ [km]',
     'P6 – Average Speed (s = v·t + s0)', 'km/h'),
    (axes567[2], s_spring, F_spring, Ffit7,
     f'k = {k_est:.4f} lbf/cm', 'Extension $s$ [cm]', 'Force $F$ [lbf]',
     'P7 – Spring Modulus (F = k·s)', 'lbf/cm'),
]:
    ax.scatter(xd, yd, color='#1565C0', zorder=5, s=60, label='Data')
    ax.plot(xd, yfit, 'r-', linewidth=2, label=f'LS fit\n{param}')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

fig567.tight_layout()
fig567.savefig("fig_p567_linreg.png", bbox_inches='tight')
plt.close(fig567)
print("\n  Figure P5-P6-P7 saved.\n")


# =============================================================================
# PROBLEM 8: Parabolic Regression  y = ax² + bx + c
# =============================================================================
#
# Basis functions: f1(x)=x², f2(x)=x, f3(x)=1
# Matrix A has columns [x², x, 1]
# β = (AᵀA)⁻¹AᵀY

print("=" * 60)
print("PROBLEM 8: Parabolic Regression (reaction time)")
print("=" * 60)

x8 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y8 = np.array([1.50, 1.28, 1.40, 1.85, 2.20])

# Design matrix
A8 = np.column_stack([x8**2, x8, np.ones_like(x8)])
beta8 = np.linalg.inv(A8.T @ A8) @ A8.T @ y8
a8, b8, c8 = beta8

print(f"  Model: y = ax² + bx + c")
print(f"  β = (AᵀA)⁻¹AᵀY")
print(f"  a = {a8:.6f}, b = {b8:.6f}, c = {c8:.6f}")

# Show system explicitly
print(f"\n  AᵀA =\n{np.round(A8.T @ A8, 2)}")
print(f"\n  AᵀY = {np.round(A8.T @ y8, 4)}")

y8_fit = A8 @ beta8
SS_res8 = np.sum((y8 - y8_fit)**2)
SS_tot8 = np.sum((y8 - np.mean(y8))**2)
R2_8 = 1 - SS_res8/SS_tot8
print(f"\n  Fitted values: {np.round(y8_fit, 4)}")
print(f"  R² = {R2_8:.6f}")

# Min of parabola: vertex at x* = -b/(2a)
x_vertex = -b8 / (2*a8)
print(f"\n  Vertex (minimum reaction time) at x* = {x_vertex:.4f} h ≈ {x_vertex:.2f} h")

x8_fine = np.linspace(0.5, 5.5, 200)
y8_fine = a8*x8_fine**2 + b8*x8_fine + c8

fig8, ax8 = plt.subplots(figsize=(7, 5))
ax8.scatter(x8, y8, color='#1565C0', s=70, zorder=5, label='Data')
ax8.plot(x8_fine, y8_fine, 'r-', linewidth=2.5,
         label=f'$\\hat{{y}} = {a8:.4f}x^2 {b8:+.4f}x {c8:+.4f}$\n$R^2={R2_8:.4f}$')
ax8.axvline(x_vertex, color='green', linestyle='--', lw=1.5,
            label=f'Min at x*={x_vertex:.2f} h')
ax8.set_title("P8 – Parabolic Regression: Reaction Time vs. Hours on Duty",
              fontweight='bold')
ax8.set_xlabel("Time on duty $x$ [h]")
ax8.set_ylabel("Reaction time $y$ [s]")
ax8.legend(fontsize=9); ax8.grid(alpha=0.3)
fig8.tight_layout()
fig8.savefig("fig_p8_parabola.png", bbox_inches='tight')
plt.close(fig8)
print("\n  Figure P8 saved.\n")


# =============================================================================
# PROBLEM 9: Generic Linear Regression  my_lin_regression(f, x, y)
# =============================================================================
#
# Builds matrix A where A[:,j] = fj(x), appends column of ones for bias.
# β = (AᵀA)⁻¹AᵀY

def my_lin_regression(f, x, y):
    """
    Generic least squares regression with arbitrary basis functions.
    f    : list of callable basis functions
    x, y : 1-D arrays of same length (data)
    Returns beta : array of length len(f)+1
                   (last element is the bias/intercept term)
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    # Build design matrix: one column per basis function + bias column
    cols = [fi(x) for fi in f] + [np.ones_like(x)]
    A = np.column_stack(cols)
    beta = np.linalg.inv(A.T @ A) @ A.T @ y
    return beta

print("=" * 60)
print("PROBLEM 9: Generic Linear Regression")
print("=" * 60)

np.random.seed(42)
x9 = np.linspace(0, 2*np.pi, 1000)
y9 = 3*np.sin(x9) - 2*np.cos(x9) + np.random.random(len(x9))
f9 = [np.sin, np.cos]

beta9 = my_lin_regression(f9, x9, y9)
print(f"  β = {np.round(beta9, 4)}")
print(f"  Expected: β[0] ≈ 3 (sin), β[1] ≈ -2 (cos), β[2] ≈ 0.5 (bias ~mean of noise)")

y9_fit = beta9[0]*np.sin(x9) + beta9[1]*np.cos(x9) + beta9[2]
SS_res9 = np.sum((y9 - y9_fit)**2)
SS_tot9 = np.sum((y9 - np.mean(y9))**2)
R2_9 = 1 - SS_res9/SS_tot9
print(f"  R² = {R2_9:.6f}")

fig9, ax9 = plt.subplots(figsize=(10, 5))
ax9.plot(x9, y9, 'b.', markersize=2, alpha=0.4, label='Noisy data')
ax9.plot(x9, y9_fit, 'r-', linewidth=2,
         label=(f'LS fit: $\\hat{{y}} = {beta9[0]:.3f}\\sin(x) '
                f'{beta9[1]:+.3f}\\cos(x) {beta9[2]:+.3f}$\n$R^2={R2_9:.4f}$'))
ax9.set_title("P9 – Generic Least Squares Regression (sin/cos basis)",
              fontweight='bold')
ax9.set_xlabel("$x$"); ax9.set_ylabel("$y$")
ax9.legend(fontsize=10); ax9.grid(alpha=0.3)
xticks = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax9.set_xticks(xticks)
ax9.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
fig9.tight_layout()
fig9.savefig("fig_p9_linreg.png", bbox_inches='tight')
plt.close(fig9)
print("  Figure P9 saved.\n")

print("=" * 60)
print("All problems complete. All figures saved to current directory.")
print("=" * 60)