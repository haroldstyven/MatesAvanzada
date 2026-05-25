"""
Fase 4 — Análisis consolidado y figuras de convergencia corregidas
Integración desde x0=0.1 donde el modo rápido ya es nulo → convergencia limpia
"""
import numpy as np
from numpy.linalg import inv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":10,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.25,"figure.dpi":150,
})

A  = np.array([[0.,1.],[-2000.,-1002.]])
c1 = 1997/998;  c2 = 997/998
x0 = 0.1;  T_end = 1.1
y0_slow = np.array([c1*np.exp(-0.2), -2*c1*np.exp(-0.2)])
exact    = lambda x: c1*np.exp(-2*x)

COLORS = {"Euler exp.":"#E24B4A","Euler mej.":"#EF9F27","RK4":"#639922",
          "Euler imp.":"#534AB7","Trapecio":"#0F6E56","Exacta":"#2C2C2A"}

def solve(name, h):
    n = int((T_end - x0) / h)
    xs = np.linspace(x0, x0+n*h, n+1)
    ys = np.zeros((n+1, 2));  ys[0] = y0_slow.copy()
    if name == "Euler exp.":
        for i in range(n): ys[i+1] = ys[i] + h*(A@ys[i])
    elif name == "Euler mej.":
        for i in range(n):
            k1=A@ys[i]; ys[i+1]=ys[i]+(h/2)*(k1+A@(ys[i]+h*k1))
    elif name == "RK4":
        for i in range(n):
            k1=A@ys[i]; k2=A@(ys[i]+(h/2)*k1)
            k3=A@(ys[i]+(h/2)*k2); k4=A@(ys[i]+h*k3)
            ys[i+1] = ys[i]+(h/6)*(k1+2*k2+2*k3+k4)
    elif name == "Euler imp.":
        M = inv(np.eye(2)-h*A)
        for i in range(n): ys[i+1] = M@ys[i]
    elif name == "Trapecio":
        I=np.eye(2); MB=inv(I-(h/2)*A)@(I+(h/2)*A)
        for i in range(n): ys[i+1] = MB@ys[i]
    ok = np.all(np.isfinite(ys)) and np.max(np.abs(ys)) < 1e12
    if not ok: return None, None
    idx = np.argmin(np.abs(xs - T_end))
    return xs, ys

def get_error(name, h):
    xs, ys = solve(name, h)
    if xs is None: return np.nan
    idx = np.argmin(np.abs(xs - T_end))
    return abs(ys[idx, 0] - exact(xs[idx]))

# ── Fig 6: Convergencia limpia log-log ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Figura 6 — Convergencia (integración desde x₀=0.1, solo modo lento)",
             fontweight="bold")

# Panel izquierdo: implícitos — h amplio donde se ve la pendiente
ax = axes[0]
ax.set_title("Métodos implícitos")
hs_imp = np.array([0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001])
for name in ["Euler imp.", "Trapecio"]:
    errs = np.array([get_error(name, h) for h in hs_imp])
    mask = ~np.isnan(errs) & (errs > 0)
    ax.loglog(hs_imp[mask], errs[mask], "o-", color=COLORS[name],
              label=name, lw=2, ms=6)

h_r = hs_imp
ax.loglog(h_r, 0.45*h_r,   "--", color="gray", lw=1, alpha=0.6, label="O(h¹)")
ax.loglog(h_r, 0.15*h_r**2,":",  color="gray", lw=1, alpha=0.6, label="O(h²)")
ax.set_xlabel("h");  ax.set_ylabel("|y_num(1.1) − y_exacta(1.1)|")
ax.legend(fontsize=9)

# Anotar pendientes
for name, p_str in [("Euler imp.","p≈1"), ("Trapecio","p≈2")]:
    errs = np.array([get_error(name, h) for h in hs_imp])
    mask = ~np.isnan(errs) & (errs > 0)
    mid  = len(hs_imp[mask]) // 2
    ax.annotate(p_str, xy=(hs_imp[mask][mid], errs[mask][mid]),
                xytext=(8,4), textcoords="offset points",
                fontsize=8, color=COLORS[name], fontweight="bold")

# Panel derecho: explícitos — zona estable h < 0.002
ax = axes[1]
ax.set_title("Métodos explícitos (h < h_max ≈ 0.002)")
hs_exp = np.array([0.002, 0.0015, 0.001, 0.0007, 0.0005, 0.0003])
for name in ["Euler exp.", "Euler mej.", "RK4"]:
    errs = np.array([get_error(name, h) for h in hs_exp])
    mask = ~np.isnan(errs) & (errs > 0)
    ax.loglog(hs_exp[mask], errs[mask], "o-", color=COLORS[name],
              label=name, lw=2, ms=6)

h_r2 = hs_exp
ax.loglog(h_r2, 0.45*h_r2,    "--", color="gray",  lw=1, alpha=0.6, label="O(h¹)")
ax.loglog(h_r2, 0.3*h_r2**2,  ":",  color="gray",  lw=1, alpha=0.6, label="O(h²)")
ax.loglog(h_r2, 30*h_r2**4,   "-.", color="brown", lw=1, alpha=0.6, label="O(h⁴)")
ax.set_xlabel("h");  ax.set_ylabel("|y_num(1.1) − y_exacta(1.1)|")
ax.legend(fontsize=9)

for name, p_str in [("Euler exp.","p≈1"),("Euler mej.","p≈2"),("RK4","p≈4")]:
    errs = np.array([get_error(name, h) for h in hs_exp])
    mask = ~np.isnan(errs) & (errs > 0)
    if mask.sum() > 1:
        mid = len(hs_exp[mask]) // 2
        ax.annotate(p_str, xy=(hs_exp[mask][mid], errs[mask][mid]),
                    xytext=(8,4), textcoords="offset points",
                    fontsize=8, color=COLORS[name], fontweight="bold")

fig.tight_layout()
fig.savefig("/Users/haroldlagares/Documents/Maestria en Ingenieria/1er semestre/Matematicas Avanzadas/Proyecto 2/graphs/fig6_convergencia_limpia.png", bbox_inches="tight")
plt.close(); print("fig6 guardada")

# ── Fig 7: Tabla visual — resumen comparativo ─────────────────────────────────
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")
fig.suptitle("Figura 7 — Tabla comparativa de métodos (Problema 5)", fontweight="bold", y=0.98)

col_labels = ["Método","Tipo","Orden","A-estable",
              "h_max","Pasos mín.\n[0,1.5]","Error (h=h_max)","Costo/paso"]
rows_data = [
    ["Euler explícito",   "Explícito", "1", "No",  "0.0020", "750",  "O(h)",   "1 eval f"],
    ["Euler mejorado",    "Explícito", "2", "No",  "0.0020", "750",  "O(h²)",  "2 evals f"],
    ["RK4",               "Explícito", "4", "No",  "0.0028", "~540", "O(h⁴)",  "4 evals f"],
    ["Adams-Bashforth 4", "Explícito", "4", "No",  "<0.001", ">1500","O(h⁴)",  "4 evals f"],
    ["Euler implícito",   "Implícito", "1", "Sí",  "∞",      "15",   "O(h)",   "1 sistema"],
    ["Trapecio (C-N)",    "Implícito", "2", "Sí",  "∞",      "15",   "O(h²)",  "1 sistema"],
    ["Adams-Moulton 4",   "Implícito", "4", "No*", "~1.5",   "~1000","O(h⁴)",  "1 sistema"],
]

colors_row = ["#FAECE7","#FAECE7","#FAECE7","#FAECE7",
              "#E1F5EE","#E1F5EE","#E6F1FB"]

table = ax.table(cellText=rows_data, colLabels=col_labels,
                 cellLoc="center", loc="center",
                 bbox=[0, 0.02, 1, 0.92])
table.auto_set_font_size(False); table.set_fontsize(9)
table.auto_set_column_width(col=list(range(len(col_labels))))

for (r,c), cell in table.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:
        cell.set_facecolor("#2C2C2A"); cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor(colors_row[r-1])

ax.text(0.01, -0.02, "* AM4 no es A-estable; región de estabilidad limitada a |λh| ≲ 6",
        transform=ax.transAxes, fontsize=7.5, color="#5F5E5A")

fig.tight_layout()
fig.savefig("/Users/haroldlagares/Documents/Maestria en Ingenieria/1er semestre/Matematicas Avanzadas/Proyecto 2/graphs/fig7_tabla_comparativa.png", bbox_inches="tight", dpi=150)
plt.close(); print("fig7 guardada")

# ── Fig 8: Costo vs precisión corregido ───────────────────────────────────────
import time as _time

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Figura 8 — Eficiencia: error vs costo computacional (desde x₀=0.1)",
             fontweight="bold")

hs_cfg = {
    "Euler exp.":  np.array([0.002,0.0015,0.001,0.0007,0.0005,0.0003]),
    "Euler mej.":  np.array([0.002,0.0015,0.001,0.0007,0.0005,0.0003]),
    "RK4":         np.array([0.002,0.0015,0.001,0.0007,0.0005,0.0003]),
    "Euler imp.":  np.array([0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.001]),
    "Trapecio":    np.array([0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.001]),
}
evals_per_step = {"Euler exp.":1,"Euler mej.":2,"RK4":4,"Euler imp.":1,"Trapecio":1}

for name, hs in hs_cfg.items():
    errs, ns, ts = [], [], []
    for h in hs:
        t0 = _time.perf_counter()
        xs, ys = solve(name, h)
        tc = _time.perf_counter() - t0
        if xs is None: continue
        idx = np.argmin(np.abs(xs - T_end))
        e = abs(ys[idx, 0] - exact(xs[idx]))
        if e > 0:
            errs.append(e); ts.append(tc)
            ns.append(len(xs)*evals_per_step[name])

    if not errs: continue
    axes[0].loglog(ns, errs, "o-", color=COLORS[name], label=name, lw=1.8, ms=5)
    axes[1].loglog(ts, errs, "o-", color=COLORS[name], label=name, lw=1.8, ms=5)

axes[0].set_xlabel("Evaluaciones de función (o soluciones de sistema)")
axes[0].set_ylabel("Error en x=1.1")
axes[0].set_title("Error vs evaluaciones")
axes[0].legend(fontsize=8)

axes[1].set_xlabel("Tiempo CPU (s)")
axes[1].set_ylabel("Error en x=1.1")
axes[1].set_title("Error vs tiempo CPU")
axes[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig("/Users/haroldlagares/Documents/Maestria en Ingenieria/1er semestre/Matematicas Avanzadas/Proyecto 2/graphs/fig8_eficiencia.png", bbox_inches="tight")
plt.close(); print("fig8 guardada")

print("\n✓ Figuras de Fase 4 generadas.")