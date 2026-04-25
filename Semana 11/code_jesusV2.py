import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

# --- Type alias ---
ScalarFunc = Callable[[float, float], float]


def rk4_step(
    f: ScalarFunc,
    t_j: float,
    s_j: float,
    h: float
) -> float:
    """
    Avanza un paso del método Runge-Kutta de 4to orden.

    Las cuatro pendientes se calculan como:
        k1 = f(t_j,          s_j)
        k2 = f(t_j + h/2,    s_j + (h/2)*k1)
        k3 = f(t_j + h/2,    s_j + (h/2)*k2)
        k4 = f(t_j + h,      s_j + h*k3)

    La actualización ponderada es:
        s_{j+1} = s_j + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    """
    k1 = f(t_j,            s_j)
    k2 = f(t_j + 0.5 * h,  s_j + 0.5 * h * k1)
    k3 = f(t_j + 0.5 * h,  s_j + 0.5 * h * k2)
    k4 = f(t_j + h,         s_j + h * k3)

    return s_j + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_rk4(
    f: ScalarFunc,
    t_span: tuple[float, float],
    s0: float,
    h: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resuelve s'(t) = f(t, s) con condición inicial s(t0) = s0
    usando RK4 con paso fijo h sobre el intervalo t_span = [t0, tf].

    Retorna:
        t_arr : array de nodos temporales
        s_arr : array de valores aproximados S(t_j)
    """
    t0, tf = t_span
    t_arr = np.arange(t0, tf, h)   # replica el t_eval original
    s_arr = np.empty_like(t_arr)
    s_arr[0] = s0

    for j in range(len(t_arr) - 1):
        s_arr[j + 1] = rk4_step(f, t_arr[j], s_arr[j], h)

    return t_arr, s_arr


# ── Definición del problema ───────────────────────────────────────────────────
F: ScalarFunc = lambda t, s: np.cos(t)   # dS/dt = cos(t)
T_SPAN        = (0.0, np.pi)
S0            = 0.0
H             = 0.1

# ── Resolución numérica con RK4 manual ───────────────────────────────────────
t, s_num = solve_rk4(F, T_SPAN, S0, H)

# ── Solución analítica exacta: S(t) = sin(t) ─────────────────────────────────
s_exact = np.sin(t)

# ── Gráficas ──────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-poster")
except OSError:
    plt.style.use("seaborn-poster")

plt.figure(figsize=(12, 4))

plt.subplot(121)
plt.plot(t, s_num)
plt.xlabel("t")
plt.ylabel("S(t)")

plt.subplot(122)
plt.plot(t, s_num - s_exact)
plt.xlabel("t")
plt.ylabel("S(t) - sin(t)")

plt.tight_layout()
plt.show()