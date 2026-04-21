"""
HW4 | Problem 2: Central Finite Difference Scheme for the IVP
    y'' + 4y' + 4y = (3+x)*exp(-2x),  y(0)=2, y'(0)=5

Author : Harold Styven Lagares De Voz  <hlagares@utb.edu.co>
Course : Maestría en Ingeniería | Matemáticas Avanzadas
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

# ──────────────────────────────────────────────────────────────────────
# 1. Exact solution (from undetermined coefficients + IVP)
# ──────────────────────────────────────────────────────────────────────
def y_exact(x):
    """y(x) = e^{-2x}(2 + 9x + (3/2)x^2 + (1/6)x^3)"""
    return np.exp(-2*x) * (2 + 9*x + 1.5*x**2 + (1/6)*x**3)

# ──────────────────────────────────────────────────────────────────────
# 2. Right-hand side f(x)
# ──────────────────────────────────────────────────────────────────────
def f(x):
    return (3 + x) * np.exp(-2*x)

# ──────────────────────────────────────────────────────────────────────
# 3. Central FD solver
#    Scheme:  y_{i+1}(1+2h) + y_i(-2+4h^2) + y_{i-1}(1-2h) = h^2 f_i
#    Rearranged:
#      y_{i+1} = [h^2 f_i + (2-4h^2) y_i - (1-2h) y_{i-1}] / (1+2h)
#
#    Starting value y_1 via ghost-point at x_0:
#      y''(x_0) = f(x_0) - 4y'(x_0) - 4y(x_0)
#      Central FD at x_0:  (y_1 - y_{-1})/(2h) = y'(x_0)
#      (y_1 - 2y_0 + y_{-1})/h^2 = y''(x_0)
#      Eliminate ghost point y_{-1} → y_1 = (y_0 + h*y'(x_0)) + h^2/2 * y''(x_0)
# ──────────────────────────────────────────────────────────────────────
def fd_solver(h, x_end=5.0):
    """Return (x_grid, y_numerical) using the central-FD scheme."""
    n = int(round(x_end / h))           # number of intervals
    x = np.linspace(0, x_end, n + 1)

    # Initial conditions
    y0  =  2.0   # y(0)
    yp0 =  5.0   # y'(0)

    # y''(x_0) from the ODE at x=0
    ypp0 = f(x[0]) - 4*yp0 - 4*y0     # = 3 - 20 - 8 = -25

    # Ghost-point formula for y_1
    y1 = y0 + h*yp0 + (h**2 / 2)*ypp0

    y = np.zeros(n + 1)
    y[0] = y0
    y[1] = y1

    # Iterative scheme for i = 1, 2, ..., n-1
    for i in range(1, n):
        num = h**2 * f(x[i]) + (2 - 4*h**2)*y[i] - (1 - 2*h)*y[i-1]
        den = 1 + 2*h
        y[i+1] = num / den

    return x, y

# ──────────────────────────────────────────────────────────────────────
# 4. Run for h = 0.5 and print table
# ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("TABLE 1 – Numerical vs. Exact solution  (h = 0.5)")
print("=" * 70)
print(f"{'i':>3}  {'x_i':>6}  {'y_i (FD)':>12}  {'y(x_i) exact':>14}  {'|error|':>12}")
print("-" * 70)

h_ref = 0.5
x_ref, y_ref = fd_solver(h_ref)
y_ex_ref = y_exact(x_ref)

for i, (xi, yi, yei) in enumerate(zip(x_ref, y_ref, y_ex_ref)):
    print(f"{i:>3}  {xi:>6.2f}  {yi:>12.6f}  {yei:>14.6f}  {abs(yi-yei):>12.6f}")

print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 5. Error convergence study: halve h from 0.5 down to 0.5/2^5
# ──────────────────────────────────────────────────────────────────────
h_values = [0.5 / 2**k for k in range(6)]   # [0.5, 0.25, 0.125, ...]
El_values = []   # local  truncation error ~ |y_2 - y(x_2)|  (first non-IC step)
Eg_values = []   # global truncation error ~ max|y_i - y(x_i)|

for h in h_values:
    x, y = fd_solver(h)
    err = np.abs(y - y_exact(x))
    El_values.append(err[2])          # error at index 2 (first iterated point)
    Eg_values.append(np.max(err))

print("\nConvergence Table")
print(f"{'h':>10}  {'E_l':>14}  {'E_g':>14}")
print("-" * 42)
for h, el, eg in zip(h_values, El_values, Eg_values):
    print(f"{h:>10.6f}  {el:>14.6e}  {eg:>14.6e}")

# Fit log-log slopes
coeff_l = np.polyfit(np.log(h_values), np.log(El_values), 1)
coeff_g = np.polyfit(np.log(h_values), np.log(Eg_values), 1)
print(f"\nLog-log slope E_l : {coeff_l[0]:.4f}  (expected ≈ 2)")
print(f"Log-log slope E_g : {coeff_g[0]:.4f}  (expected ≈ 2)")

# ──────────────────────────────────────────────────────────────────────
# 6. Figure 1 – Truncation error log-log plot
# ──────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.loglog(h_values, El_values, 'o-', color='steelblue',  label=r'Local error $E_l$')
ax1.loglog(h_values, Eg_values, 's-', color='darkorange', label=r'Global error $E_g$')

# Reference O(h^2) line
h_arr = np.array(h_values)
ax1.loglog(h_arr, Eg_values[0]*(h_arr/h_arr[0])**2, 'k--', alpha=0.5,
           label=r'$O(h^2)$ reference')

ax1.set_xlabel('Step size $h$', fontsize=12)
ax1.set_ylabel('Truncation error', fontsize=12)
ax1.set_title('Convergence of Central FD Scheme\n'
              r'$y''+4y\'+4y=(3+x)e^{-2x}$', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)
ax1.annotate(f'slope $E_l$ ≈ {coeff_l[0]:.2f}', xy=(0.05, 0.15),
             xycoords='axes fraction', fontsize=10, color='steelblue')
ax1.annotate(f'slope $E_g$ ≈ {coeff_g[0]:.2f}', xy=(0.05, 0.08),
             xycoords='axes fraction', fontsize=10, color='darkorange')
fig1.tight_layout()
fig1.savefig('truncation_errors.png', dpi=150)
print("\nSaved: truncation_errors.png")

# ──────────────────────────────────────────────────────────────────────
# 7. Figure 2 – Solution comparison (exact vs FD, h=0.5 and h=0.0625)
# ──────────────────────────────────────────────────────────────────────
x_fine = np.linspace(0, 5, 500)
y_fine = y_exact(x_fine)

x_h1, y_h1 = fd_solver(0.5)
x_h2, y_h2 = fd_solver(0.0625)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(x_fine, y_fine, 'k-',  lw=2,   label='Exact solution')
ax2.plot(x_h1,  y_h1,  'o--',  color='steelblue',  ms=6, label='FD  $h=0.5$')
ax2.plot(x_h2,  y_h2,  's-',   color='darkorange',  ms=3, label='FD  $h=0.0625$')
ax2.set_xlabel('$x$', fontsize=12)
ax2.set_ylabel('$y$', fontsize=12)
ax2.set_title('Exact vs. Central FD Solution\n'
              r'$y''+4y\'+4y=(3+x)e^{-2x}$,  $y(0)=2$, $y\'(0)=5$', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.5)
fig2.tight_layout()
fig2.savefig('solution_comparison.png', dpi=150)
print("Saved: solution_comparison.png")

plt.show()
print("\nDone.")