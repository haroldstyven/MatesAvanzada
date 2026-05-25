"""
Fase 3 — Experimentos numéricos y gráficas
Problema 5: y'' + 1002y' + 2000y = 0, y(0)=3, y'(0)=-2000
"""

import os
import numpy as np
from numpy.linalg import inv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# Asegurar que el directorio de gráficas exista
os.makedirs("/Users/haroldlagares/Documents/Maestria en Ingenieria/1er semestre/Matematicas Avanzadas/Proyecto 2/graphs", exist_ok=True)

# ── Configuración visual consistente ─────────────────────────────────────────
COLORS = {
    "Euler exp.":    "#E24B4A",
    "Euler mej.":    "#EF9F27",
    "RK4":           "#639922",
    "AB4":           "#D85A30",
    "Euler imp.":    "#534AB7",
    "Trapecio":      "#0F6E56",
    "AM4":           "#185FA5",
    "Exacta":        "#2C2C2A",
}
LS = {  # line styles
    "Euler exp.": "--", "Euler mej.": "--", "RK4": "--", "AB4": "--",
    "Euler imp.": "-",  "Trapecio":   "-",  "AM4": "-",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 150,
})

# ── Problema ─────────────────────────────────────────────────────────────────
A   = np.array([[0., 1.], [-2000., -1002.]])
y0  = np.array([3., -2000.])
T   = 1.5
c1  = 1997/998;  c2 = 997/998

def y_exact(x):
    return c1*np.exp(-2*x) + c2*np.exp(-1000*x)

def exact_vec(x):
    return np.array([c1*np.exp(-2*x)+c2*np.exp(-1000*x),
                    -2*c1*np.exp(-2*x)-1000*c2*np.exp(-1000*x)])

# ── Solvers ───────────────────────────────────────────────────────────────────
def _rk4_boot(ys, xs, h, steps):
    for i in range(steps):
        k1=A@ys[i]; k2=A@(ys[i]+(h/2)*k1)
        k3=A@(ys[i]+(h/2)*k2); k4=A@(ys[i]+h*k3)
        ys[i+1]=ys[i]+(h/6)*(k1+2*k2+2*k3+k4)

def solve(name, h):
    n=int(T/h); xs=np.linspace(0,n*h,n+1)
    ys=np.zeros((n+1,2)); ys[0]=y0.copy()
    if name=="Euler exp.":
        for i in range(n): ys[i+1]=ys[i]+h*(A@ys[i])
    elif name=="Euler mej.":
        for i in range(n):
            k1=A@ys[i]; ys[i+1]=ys[i]+(h/2)*(k1+A@(ys[i]+h*k1))
    elif name=="RK4":
        for i in range(n):
            k1=A@ys[i]; k2=A@(ys[i]+(h/2)*k1)
            k3=A@(ys[i]+(h/2)*k2); k4=A@(ys[i]+h*k3)
            ys[i+1]=ys[i]+(h/6)*(k1+2*k2+2*k3+k4)
    elif name=="AB4":
        _rk4_boot(ys,xs,h,min(3,n))
        for i in range(3,n):
            f0=A@ys[i];f1=A@ys[i-1];f2=A@ys[i-2];f3=A@ys[i-3]
            ys[i+1]=ys[i]+(h/24)*(55*f0-59*f1+37*f2-9*f3)
    elif name=="Euler imp.":
        M=inv(np.eye(2)-h*A)
        for i in range(n): ys[i+1]=M@ys[i]
    elif name=="Trapecio":
        I=np.eye(2); MB=inv(I-(h/2)*A)@(I+(h/2)*A)
        for i in range(n): ys[i+1]=MB@ys[i]
    elif name=="AM4":
        _rk4_boot(ys,xs,h,min(3,n))
        M=inv(np.eye(2)-(9*h/24)*A)
        for i in range(3,n):
            f0=A@ys[i];f1=A@ys[i-1];f2=A@ys[i-2]
            ys[i+1]=M@(ys[i]+(h/24)*(19*f0-5*f1+f2))
    stable=np.all(np.isfinite(ys)) and np.max(np.abs(ys))<1e12
    return xs, ys, stable

# ── FIGURA 1: Estabilidad — explícito vs implícito con h moderado ─────────────
def fig_estabilidad():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Figura 1 — Estabilidad: explícito vs implícito  (h = 0.01)", fontweight="bold")

    xs_ex = np.linspace(0, T, 500)
    ys_ex = y_exact(xs_ex)

    # Panel izquierdo: métodos explícitos con h=0.01 (inestable)
    ax = axes[0]
    ax.set_title("Métodos explícitos — h = 0.01 (inestable)")
    for name in ["Euler exp.", "Euler mej.", "RK4"]:
        xs, ys, _ = solve(name, 0.01)
        # Truncar para graficar hasta donde explota
        mask = np.abs(ys[:,0]) < 50
        cut = np.argmax(~mask) if not mask.all() else len(xs)
        cut = min(cut+2, len(xs))
        ax.plot(xs[:cut], ys[:cut,0], LS[name], color=COLORS[name],
                label=name, lw=1.5, alpha=0.85)
    ax.plot(xs_ex, ys_ex, "-", color=COLORS["Exacta"], lw=2, label="Exacta", zorder=5)
    ax.set_xlim(0, 0.15); ax.set_ylim(-50, 10)
    ax.set_xlabel("x"); ax.set_ylabel("y(x)")
    ax.legend(fontsize=8); ax.axhline(0, color="gray", lw=0.5)

    # Panel derecho: métodos implícitos con h=0.01 (estables)
    ax = axes[1]
    ax.set_title("Métodos implícitos — h = 0.01 (estables)")
    for name in ["Euler imp.", "Trapecio"]:
        xs, ys, _ = solve(name, 0.01)
        ax.plot(xs, ys[:,0], LS[name], color=COLORS[name],
                label=name, lw=1.8, alpha=0.9)
    ax.plot(xs_ex, ys_ex, "-", color=COLORS["Exacta"], lw=2, label="Exacta", zorder=5)
    ax.set_xlim(0, T); ax.set_xlabel("x"); ax.set_ylabel("y(x)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("/Users/haroldlagares/Documents/Maestria en Ingenieria/1er semestre/Matematicas Avanzadas/Proyecto 2/graphs/fig1_estabilidad.png", bbox_inches="tight")
    plt.close()
    print("fig1_estabilidad.png guardada")

# ── FIGURA 2: Solución completa con h pequeño — todos los métodos estables ───
def fig_solucion():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Figura 2 — Solución con h = 0.001 (todos los métodos estables)", fontweight="bold")

    xs_ex = np.linspace(0, T, 2000)
    ys_ex = y_exact(xs_ex)

    # Panel izquierdo: intervalo completo
    ax = axes[0]
    ax.set_title("Intervalo completo [0, 1.5]")
    ax.plot(xs_ex, ys_ex, "-", color=COLORS["Exacta"], lw=2.5, label="Exacta", zorder=5)
    for name in ["Euler imp.", "Trapecio", "RK4"]:
        xs, ys, ok = solve(name, 0.001)
        if ok:
            ax.plot(xs[::3], ys[::3,0], LS[name], color=COLORS[name],
                    label=name, lw=1.3, alpha=0.75)
    ax.set_xlabel("x"); ax.set_ylabel("y(x)")
    ax.legend(fontsize=8)

    # Panel derecho: zoom en zona transitoria
    ax = axes[1]
    ax.set_title("Zoom zona transitoria [0, 0.008]")
    xs_ez = np.linspace(0, 0.008, 500)
    ax.plot(xs_ez, y_exact(xs_ez), "-", color=COLORS["Exacta"], lw=2.5, label="Exacta")
    for name in ["Euler exp.", "Euler imp.", "Trapecio"]:
        xs, ys, ok = solve(name, 0.001)
        if ok:
            mask = xs <= 0.008
            ax.plot(xs[mask], ys[mask,0], LS[name], color=COLORS[name],
                    label=name, lw=1.5, alpha=0.85)
    ax.set_xlabel("x"); ax.set_ylabel("y(x)")
    ax.legend(fontsize=8)
    ax.set_title("Zoom zona transitoria [0, 0.008]\n(modo rápido e⁻¹⁰⁰⁰ˣ)")

    fig.tight_layout()
    fig.savefig("/Users/haroldlagares/Documents/Maestria en Ingenieria/1er semestre/Matematicas Avanzadas/Proyecto 2/graphs/fig2_solucion.png", bbox_inches="tight")
    plt.close()
    print("fig2_solucion.png guardada")

# ── FIGURA 3: Convergencia — log(error) vs log(h) ────────────────────────────
def fig_convergencia():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Figura 3 — Convergencia: log(error) vs log(h)  [error en T=1, modo lento]",
                 fontweight="bold")

    # Medimos en T=1 comparando contra c1*exp(-2) (modo rapido ya nulo)
    exact_T1 = c1 * np.exp(-2.0)

    def err_at_T1(name, h):
        # Integrar solo hasta T=1
        global T
        T_orig = T; T = 1.0
        xs, ys, ok = solve(name, h)
        T = T_orig
        if not ok: return np.nan
        idx = np.argmin(np.abs(xs - 1.0))
        return abs(ys[idx, 0] - exact_T1)

    # Implícitos: rango h grande donde se ve la pendiente
    ax = axes[0]
    ax.set_title("Métodos implícitos")
    hs_imp = np.array([0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001])
    for name in ["Euler imp.", "Trapecio"]:
        errs = [err_at_T1(name, h) for h in hs_imp]
        errs = np.array(errs)
        mask = ~np.isnan(errs) & (errs > 0)
        ax.loglog(hs_imp[mask], errs[mask], "o-", color=COLORS[name],
                  label=name, lw=1.8, ms=5)
    # Líneas de referencia O(h) y O(h^2)
    h_ref = hs_imp[mask]
    ax.loglog(h_ref, 0.15*h_ref,    "--", color="gray", lw=1, alpha=0.6, label="O(h)")
    ax.loglog(h_ref, 0.05*h_ref**2, ":",  color="gray", lw=1, alpha=0.6, label="O(h²)")
    ax.set_xlabel("h"); ax.set_ylabel("Error global |y_num − y_exacta|")
    ax.legend(fontsize=8)

    # Explícitos: rango h < h_max = 0.002
    ax = axes[1]
    ax.set_title("Métodos explícitos (h < h_max ≈ 0.002)")
    hs_exp = np.array([0.0018, 0.0014, 0.0010, 0.0007, 0.0005, 0.0003])
    for name in ["Euler exp.", "Euler mej.", "RK4"]:
        errs = [err_at_T1(name, h) for h in hs_exp]
        errs = np.array(errs)
        mask = ~np.isnan(errs) & (errs > 0)
        ax.loglog(hs_exp[mask], errs[mask], "o-", color=COLORS[name],
                  label=name, lw=1.8, ms=5)
    h_ref2 = hs_exp
    ax.loglog(h_ref2, 0.15*h_ref2,    "--", color="gray", lw=1, alpha=0.6, label="O(h)")
    ax.loglog(h_ref2, 0.05*h_ref2**2, ":",  color="gray", lw=1, alpha=0.6, label="O(h²)")
    ax.loglog(h_ref2, 0.002*h_ref2**4,":",  color="brown",lw=1, alpha=0.6, label="O(h⁴)")
    ax.set_xlabel("h"); ax.set_ylabel("Error global")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("/Users/haroldlagares/Documents/Maestria en Ingenieria/1er semestre/Matematicas Avanzadas/Proyecto 2/graphs/fig3_convergencia.png", bbox_inches="tight")
    plt.close()
    print("fig3_convergencia.png guardada")

# ── FIGURA 4: Costo computacional ────────────────────────────────────────────
def fig_costo():
    import time as _time
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Figura 4 — Costo computacional vs precisión", fontweight="bold")

    exact_T1 = c1 * np.exp(-2.0)
    global T
    T_orig = T; T = 1.0

    resultados = []
    hs_test = {
        "Euler exp.":  [0.0018, 0.0014, 0.001, 0.0007, 0.0005],
        "RK4":         [0.0018, 0.0014, 0.001, 0.0007, 0.0005],
        "Euler imp.":  [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
        "Trapecio":    [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
    }
    for name, hs in hs_test.items():
        for h in hs:
            t0=_time.perf_counter()
            xs,ys,ok=solve(name,h)
            tc=_time.perf_counter()-t0
            if ok:
                idx=np.argmin(np.abs(xs-1.0))
                err=abs(ys[idx,0]-exact_T1)
                n_pasos=len(xs)-1
                resultados.append((name,h,n_pasos,tc,err))

    T = T_orig

    # Panel 1: error vs n_pasos
    ax = axes[0]
    ax.set_title("Error vs número de pasos")
    for name in ["Euler exp.","RK4","Euler imp.","Trapecio"]:
        pts = [(r[2],r[4]) for r in resultados if r[0]==name and r[4]>0]
        if pts:
            ns,es=zip(*pts)
            ax.loglog(ns,es,"o-",color=COLORS[name],label=name,lw=1.8,ms=5)
    ax.set_xlabel("Número de pasos"); ax.set_ylabel("Error en T=1")
    ax.legend(fontsize=8)

    # Panel 2: error vs tiempo CPU
    ax = axes[1]
    ax.set_title("Error vs tiempo CPU (s)")
    for name in ["Euler exp.","RK4","Euler imp.","Trapecio"]:
        pts = [(r[3],r[4]) for r in resultados if r[0]==name and r[4]>0]
        if pts:
            ts,es=zip(*pts)
            ax.loglog(ts,es,"o-",color=COLORS[name],label=name,lw=1.8,ms=5)
    ax.set_xlabel("Tiempo CPU (s)"); ax.set_ylabel("Error en T=1")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("/Users/haroldlagares/Documents/Maestria en Ingenieria/1er semestre/Matematicas Avanzadas/Proyecto 2/graphs/fig4_costo.png", bbox_inches="tight")
    plt.close()
    print("fig4_costo.png guardada")

# ── FIGURA 5: h_max empírico ──────────────────────────────────────────────────
def fig_hmax():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_title("Figura 5 — Búsqueda empírica del h_max (métodos explícitos)", fontweight="bold")

    hs_scan = np.logspace(-4, -1.5, 60)
    for name in ["Euler exp.", "Euler mej.", "RK4", "AB4"]:
        stabs = []
        for h in hs_scan:
            xs,ys,ok=solve(name, h)
            stabs.append(1 if ok else 0)
        stabs=np.array(stabs)
        # h_max = mayor h estable
        idx_max = np.where(stabs==1)[0]
        h_emp = hs_scan[idx_max[-1]] if len(idx_max)>0 else np.nan
        ax.semilogx(hs_scan, stabs+0.05*["Euler exp.","Euler mej.","RK4","AB4"].index(name),
                    "-", color=COLORS[name], lw=2, label=f"{name}  h_max≈{h_emp:.4f}", alpha=0.8)
        if not np.isnan(h_emp):
            ax.axvline(h_emp, color=COLORS[name], lw=1, ls="--", alpha=0.5)

    ax.axvline(2/1000, color="black", lw=1.5, ls=":", label="Teórico 2/|λ₂|=0.002")
    ax.set_xlabel("h"); ax.set_ylabel("Estable (1) / Inestable (0)")
    ax.set_yticks([0,1]); ax.set_yticklabels(["Inestable","Estable"])
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig("/Users/haroldlagares/Documents/Maestria en Ingenieria/1er semestre/Matematicas Avanzadas/Proyecto 2/graphs/fig5_hmax.png", bbox_inches="tight")
    plt.close()
    print("fig5_hmax.png guardada")

# ── Ejecutar todo ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generando figuras de Fase 3...")
    fig_estabilidad()
    fig_solucion()
    fig_convergencia()
    fig_costo()
    fig_hmax()
    print("\n✓ Todas las figuras guardadas en /Users/haroldlagares/Documents/Maestria en Ingenieria/1er semestre/Matematicas Avanzadas/Proyecto 2/graphs")