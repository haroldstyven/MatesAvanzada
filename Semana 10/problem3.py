"""
HW4 | Problem 3: Projectile Motion (no air resistance)
    x'' = 0,   y'' = -g
    Recast as 1st-order system:  s = [x, y, vx, vy]
    s' = [vx, vy, 0, -g]

Author : Harold Styven Lagares De Voz  <hlagares@utb.edu.co>
Course : Maestría en Ingeniería | Matemáticas Avanzadas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ──────────────────────────────────────────────────────────────────────
# 1. Solver function
# ──────────────────────────────────────────────────────────────────────
def my_projectile_motion_solver(t_span, s0, m, g):
    """
    Solve the 2-D projectile motion (no air resistance).

    Parameters
    ----------
    t_span : list [t0, tf]   | time interval
    s0     : list [x0, y0, vx0, vy0]  | initial state (position + velocity)
    m      : float | mass of the projectile (kg)  [not used; Newton: ma=-mg → a=-g]
    g      : float | gravitational acceleration (m/s²)

    Returns
    -------
    T : 1-D array of times
    X : 1-D array of x-positions
    Y : 1-D array of y-positions
    """
    y0_launch = s0[1]   # initial height, used as ground reference

    # ODE right-hand side:  s = [x, y, vx, vy]
    def ode(t, s):
        x, y, vx, vy = s
        return [vx, vy, 0.0, -g]

    # Terminal event: projectile returns to launch height
    def hit_ground(t, s):
        return s[1] - y0_launch          # zero when y = y0_launch

    hit_ground.terminal  = True
    hit_ground.direction = -1            # trigger only on descent

    sol = solve_ivp(
        ode,
        t_span,
        s0,
        method='RK45',
        events=hit_ground,
        dense_output=True,
        max_step=(t_span[1] - t_span[0]) / 1000
    )

    T = sol.t
    X = sol.y[0]
    Y = sol.y[1]
    return T, X, Y

# ──────────────────────────────────────────────────────────────────────
# 2. Analytical solution (for verification)
# ──────────────────────────────────────────────────────────────────────
def projectile_exact(t, s0, g):
    x0, y0, vx0, vy0 = s0
    x = x0 + vx0 * t
    y = y0 + vy0 * t - 0.5 * g * t**2
    return x, y

# ──────────────────────────────────────────────────────────────────────
# 3. Demo run: 45° launch, v0 = 50 m/s, launch from ground (0,0)
# ──────────────────────────────────────────────────────────────────────
g  = 9.81          # m/s²
m  = 1.0           # kg  (irrelevant for trajectory without drag)
v0 = 50.0          # m/s
theta = 45.0       # degrees

vx0 = v0 * np.cos(np.radians(theta))
vy0 = v0 * np.sin(np.radians(theta))
s0  = [0.0, 0.0, vx0, vy0]
t_span = [0.0, 2 * vy0 / g + 1.0]   # generous upper bound

T, X, Y = my_projectile_motion_solver(t_span, s0, m, g)

# Exact reference
T_ex  = np.linspace(0, 2*vy0/g, 500)
X_ex, Y_ex = projectile_exact(T_ex, s0, g)

print("=" * 60)
print(f"Projectile demo: v0={v0} m/s, θ={theta}°, g={g} m/s²")
print(f"  Range   (RK45)  : {X[-1]:.4f} m")
print(f"  Range   (exact) : {vx0*2*vy0/g:.4f} m")
print(f"  Max height (RK45)  : {max(Y):.4f} m")
print(f"  Max height (exact) : {vy0**2/(2*g):.4f} m")
print(f"  Flight time (RK45) : {T[-1]:.4f} s")
print(f"  Flight time (exact): {2*vy0/g:.4f} s")
print("=" * 60)

# ──────────────────────────────────────────────────────────────────────
# 4. Multiple launch angles
# ──────────────────────────────────────────────────────────────────────
angles = [15, 30, 45, 60, 75]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(angles)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Left panel: trajectories ---
ax = axes[0]
for ang, col in zip(angles, colors):
    vx_ = v0 * np.cos(np.radians(ang))
    vy_ = v0 * np.sin(np.radians(ang))
    s_  = [0.0, 0.0, vx_, vy_]
    T_, X_, Y_ = my_projectile_motion_solver([0, 2*vy_/g+1], s_, m, g)
    ax.plot(X_, Y_, color=col, lw=2, label=f'θ = {ang}°')

ax.set_xlabel('Horizontal distance $x$ (m)', fontsize=11)
ax.set_ylabel('Vertical height $y$ (m)', fontsize=11)
ax.set_title(f'Projectile Trajectories\n$v_0={v0}$ m/s, no air resistance', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_ylim(bottom=0)

# --- Right panel: 45° comparison with exact ---
ax2 = axes[1]
ax2.plot(X_ex, Y_ex, 'k-',  lw=2, label='Analytical solution')
ax2.plot(X,    Y,    'o--', color='steelblue', ms=4, markevery=10,
         label='RK45 (solve_ivp)')
ax2.set_xlabel('Horizontal distance $x$ (m)', fontsize=11)
ax2.set_ylabel('Vertical height $y$ (m)', fontsize=11)
ax2.set_title(f'RK45 vs. Analytical (θ=45°, $v_0={v0}$ m/s)', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_ylim(bottom=0)

fig.tight_layout()
fig.savefig('projectile_motion.png', dpi=150)
print("\nSaved: projectile_motion.png")
plt.show()
print("Done.")