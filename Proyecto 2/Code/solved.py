"""
Problema 5 - Rigidez extrema
y'' + 1002y' + 2000y = 0,  y(0)=3,  y'(0)=-2000
Sistema: y' = Ay,  A = [[0,1],[-2000,-1002]],  y0 = [3,-2000]
Autovalores: lambda1=-2, lambda2=-1000  (S=500)
"""

import numpy as np
import time

# ─────────────────────────────────────────
#  Definición del problema
# ─────────────────────────────────────────
A  = np.array([[0.0, 1.0], [-2000.0, -1002.0]])
y0 = np.array([3.0, -2000.0])
T  = 1.5                          # intervalo [0, T]

c1 = 1997 / 998
c2 =  997 / 998

def f(x, y):
    """RHS del sistema: f(x,y) = A @ y  (sin término fuente)"""
    return A @ y

def y_exact(x):
    """Solución exacta escalar y(x)"""
    return c1 * np.exp(-2.0 * x) + c2 * np.exp(-1000.0 * x)

def y_exact_vec(x):
    """Solución exacta vectorial [y(x), y'(x)]"""
    return np.array([
        c1 * np.exp(-2.0 * x)  + c2 * np.exp(-1000.0 * x),
       -2*c1 * np.exp(-2.0 * x) - 1000*c2 * np.exp(-1000.0 * x)
    ])

# ─────────────────────────────────────────
#  Métodos EXPLÍCITOS
# ─────────────────────────────────────────

def euler_explicito(h):
    """Euler explícito: y_{n+1} = y_n + h*f(x_n, y_n)"""
    n  = int(T / h)
    xs = np.linspace(0, n * h, n + 1)
    ys = np.zeros((n + 1, 2))
    ys[0] = y0.copy()
    evals = 0
    for i in range(n):
        ys[i+1] = ys[i] + h * f(xs[i], ys[i])
        evals += 1
    return xs, ys, evals


def euler_mejorado(h):
    """Euler mejorado / Heun: predictor-corrector explícito O(h^2)"""
    n  = int(T / h)
    xs = np.linspace(0, n * h, n + 1)
    ys = np.zeros((n + 1, 2))
    ys[0] = y0.copy()
    evals = 0
    for i in range(n):
        k1 = f(xs[i],     ys[i])
        k2 = f(xs[i] + h, ys[i] + h * k1)
        ys[i+1] = ys[i] + (h / 2) * (k1 + k2)
        evals += 2
    return xs, ys, evals


def rk4(h):
    """Runge-Kutta clásico de 4° orden"""
    n  = int(T / h)
    xs = np.linspace(0, n * h, n + 1)
    ys = np.zeros((n + 1, 2))
    ys[0] = y0.copy()
    evals = 0
    for i in range(n):
        k1 = f(xs[i],           ys[i])
        k2 = f(xs[i] + h/2,     ys[i] + (h/2)*k1)
        k3 = f(xs[i] + h/2,     ys[i] + (h/2)*k2)
        k4 = f(xs[i] + h,       ys[i] + h*k3)
        ys[i+1] = ys[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        evals += 4
    return xs, ys, evals


def adams_bashforth(h):
    """Adams-Bashforth 4 pasos (explícito, O(h^4)).
    Arranque con RK4 para los 3 primeros puntos."""
    n  = int(T / h)
    xs = np.linspace(0, n * h, n + 1)
    ys = np.zeros((n + 1, 2))
    ys[0] = y0.copy()
    evals = 0

    # Arranque RK4
    for i in range(min(3, n)):
        k1 = f(xs[i],       ys[i])
        k2 = f(xs[i]+h/2,   ys[i]+(h/2)*k1)
        k3 = f(xs[i]+h/2,   ys[i]+(h/2)*k2)
        k4 = f(xs[i]+h,     ys[i]+h*k3)
        ys[i+1] = ys[i] + (h/6)*(k1+2*k2+2*k3+k4)
        evals += 4

    # AB4 a partir del paso 3
    for i in range(3, n):
        f0 = f(xs[i],   ys[i])
        f1 = f(xs[i-1], ys[i-1])
        f2 = f(xs[i-2], ys[i-2])
        f3 = f(xs[i-3], ys[i-3])
        ys[i+1] = ys[i] + (h/24)*(55*f0 - 59*f1 + 37*f2 - 9*f3)
        evals += 4
    return xs, ys, evals


# ─────────────────────────────────────────
#  Métodos IMPLÍCITOS
#  Para A constante se precalcula la inversa
#  una sola vez → eficiencia O(1) por paso
# ─────────────────────────────────────────

def euler_implicito(h):
    """Backward Euler: y_{n+1} = (I - hA)^{-1} y_n
    A-estable: acepta cualquier h."""
    n   = int(T / h)
    xs  = np.linspace(0, n * h, n + 1)
    ys  = np.zeros((n + 1, 2))
    ys[0] = y0.copy()
    M_inv = np.linalg.inv(np.eye(2) - h * A)   # precalculada
    solves = 0
    for i in range(n):
        ys[i+1] = M_inv @ ys[i]
        solves += 1
    return xs, ys, solves


def trapecio(h):
    """Crank-Nicolson / Trapecio: O(h^2), A-estable.
    y_{n+1} = (I - h/2·A)^{-1} (I + h/2·A) y_n"""
    n   = int(T / h)
    xs  = np.linspace(0, n * h, n + 1)
    ys  = np.zeros((n + 1, 2))
    ys[0] = y0.copy()
    I   = np.eye(2)
    M_inv = np.linalg.inv(I - (h/2) * A)
    B     = I + (h/2) * A                       # ambas precalculadas
    MB    = M_inv @ B
    solves = 0
    for i in range(n):
        ys[i+1] = MB @ ys[i]
        solves += 1
    return xs, ys, solves


def adams_moulton(h):
    """Adams-Moulton 4 pasos (implícito, O(h^4)).
    Predictor Adams-Bashforth 4 + corrector AM4.
    Para A lineal el corrector es directo (un sistema lineal por paso)."""
    n  = int(T / h)
    xs = np.linspace(0, n * h, n + 1)
    ys = np.zeros((n + 1, 2))
    ys[0] = y0.copy()
    solves = 0

    # Arranque RK4
    for i in range(min(3, n)):
        k1 = f(xs[i],       ys[i])
        k2 = f(xs[i]+h/2,   ys[i]+(h/2)*k1)
        k3 = f(xs[i]+h/2,   ys[i]+(h/2)*k2)
        k4 = f(xs[i]+h,     ys[i]+h*k3)
        ys[i+1] = ys[i] + (h/6)*(k1+2*k2+2*k3+k4)

    # Precalcular inversa para el corrector AM4
    # AM4: y_{n+1} = y_n + h/24*(9*f_{n+1} + 19*f_n - 5*f_{n-1} + f_{n-2})
    # Para f=Ay:  (I - 9h/24·A) y_{n+1} = y_n + h/24*(19*Ay_n - 5*Ay_{n-1} + Ay_{n-2})
    M_inv_am = np.linalg.inv(np.eye(2) - (9*h/24) * A)

    for i in range(3, n):
        # Predictor AB4
        f0=f(xs[i],   ys[i]);   f1=f(xs[i-1],ys[i-1])
        f2=f(xs[i-2], ys[i-2]); f3=f(xs[i-3],ys[i-3])
        yp = ys[i] + (h/24)*(55*f0 - 59*f1 + 37*f2 - 9*f3)

        # Corrector AM4 (exacto para sistema lineal)
        rhs = ys[i] + (h/24)*(19*f0 - 5*f1 + f2)
        ys[i+1] = M_inv_am @ rhs
        solves += 1
    return xs, ys, solves


# ─────────────────────────────────────────
#  Utilidades de error y convergencia
# ─────────────────────────────────────────

def error_global(xs, ys):
    """Error global máximo en y (componente 0) respecto a solución exacta.
    Se omite la zona transitoria x < 0.05 donde e^{-1000x} domina."""
    errs = [abs(ys[i, 0] - y_exact(xs[i])) for i in range(len(xs)) if xs[i] >= 0.05]
    return max(errs) if errs else float('inf')

def error_final(xs, ys):
    """Error en el punto final T."""
    return abs(ys[-1, 0] - y_exact(xs[-1]))

def run_method(method, h, label=""):
    """Ejecuta un método, mide tiempo y devuelve métricas."""
    t0 = time.perf_counter()
    xs, ys, ops = method(h)
    elapsed = time.perf_counter() - t0

    # Chequeo de estabilidad: ¿explota la solución?
    stable = np.all(np.isfinite(ys)) and np.max(np.abs(ys)) < 1e10

    eg = error_global(xs, ys) if stable else float('inf')
    ef = error_final(xs, ys)  if stable else float('inf')

    return {
        "metodo":  label or method.__name__,
        "h":       h,
        "n_pasos": len(xs) - 1,
        "ops":     ops,
        "t_cpu":   elapsed,
        "estable": stable,
        "e_global": eg,
        "e_final":  ef,
    }


if __name__ == "__main__":
    import pandas as pd

    # ── Tabla comparativa con tres tamaños de paso ──────────────────────────
    metodos_exp = [
        (euler_explicito,  "Euler explícito"),
        (euler_mejorado,   "Euler mejorado"),
        (rk4,              "RK4"),
        (adams_bashforth,  "Adams-Bashforth 4"),
    ]
    metodos_imp = [
        (euler_implicito,  "Euler implícito"),
        (trapecio,         "Trapecio (C-N)"),
        (adams_moulton,    "Adams-Moulton 4"),
    ]

    pasos = {
        "h pequeño (0.001)":  0.001,
        "h mediano (0.01)":   0.01,
        "h grande (0.1)":     0.1,
    }

    rows = []
    for nombre_h, h in pasos.items():
        for met, lbl in metodos_exp + metodos_imp:
            r = run_method(met, h, lbl)
            r["caso"] = nombre_h
            rows.append(r)

    df = pd.DataFrame(rows)[["caso","metodo","h","n_pasos","ops","estable","e_global","t_cpu"]]
    pd.set_option("display.float_format", "{:.2e}".format)
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 120)
    print(df.to_string(index=False))

    # ── Verificación de órdenes de convergencia (error en T=1.5) ────────────
    print("\n── Orden de convergencia — error en x=T=1.5 (métodos implícitos) ──")
    hs_imp = [0.1, 0.05, 0.02, 0.01, 0.005]
    for met, lbl in metodos_imp:
        errs = []
        for h in hs_imp:
            r = run_method(met, h)
            errs.append(r["e_final"] if r["estable"] else np.nan)
        orders = []
        for i in range(1, len(hs_imp)):
            if not np.isnan(errs[i]) and not np.isnan(errs[i-1]) and errs[i] > 0 and errs[i-1] > 0:
                p = np.log(errs[i-1]/errs[i]) / np.log(hs_imp[i-1]/hs_imp[i])
                orders.append(round(p, 2))
        print(f"  {lbl:22s}: errores={[f'{e:.2e}' for e in errs]}  →  órdenes={orders}")

    print("\n── Orden de convergencia — error en x=T=1.5 (métodos explícitos) ──")
    hs_exp = [0.0015, 0.001, 0.0007, 0.0005, 0.0003]
    for met, lbl in metodos_exp:
        errs = []
        for h in hs_exp:
            r = run_method(met, h)
            errs.append(r["e_final"] if r["estable"] else np.nan)
        orders = []
        for i in range(1, len(hs_exp)):
            if not np.isnan(errs[i]) and not np.isnan(errs[i-1]) and errs[i] > 0 and errs[i-1] > 0:
                p = np.log(errs[i-1]/errs[i]) / np.log(hs_exp[i-1]/hs_exp[i])
                orders.append(round(p, 2))
        print(f"  {lbl:22s}: errores={[f'{e:.2e}' for e in errs]}  →  órdenes={orders}")