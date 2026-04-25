import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp # Para la gráfica de error original
import pandas as pd

# Usar un estilo de Matplotlib similar al de la imagen
try:
    plt.style.use("seaborn-v0_8-poster")
except:
    # Fallback si el estilo 'seaborn-v0_8-poster' no está disponible
    pass

# 1. Definir la función ODE (derivada) y la solución exacta
def F(t, S):
    """Define la derivada dS/dt = cos(t)"""
    return np.cos(t)

def exact_solution(t):
    """Define la solución exacta S(t) = sin(t) (dado S_0 = 0)"""
    return np.sin(t)

# Configuración de los parámetros para ambos métodos
h = 0.1 # Tamaño de paso
t_start = 0
t_end = np.pi
num_points = int(t_end / h) + 1 # Calcular número de puntos para incluir t_end
t_values = np.linspace(t_start, t_end, num_points) # Puntos de tiempo para ambas soluciones
S0 = 0 # Condición inicial S(0) = 0

# =========================================================================
# SOLUCIÓN USANDO SCIPY.INTEGRATE.SOLVE_IVP (PARA REPRODUCIR LA GRÁFICA ORIGINAL DEL ERROR)
# =========================================================================

# Resolver la ODE usando solve_ivp (utiliza RK45 por defecto)
sol_ivp = solve_ivp(F, [t_start, t_end], [S0], t_eval=t_values)

# Calcular solución exacta y error para la solución de solve_ivp
S_exact_ivp = exact_solution(sol_ivp.t)
error_ivp = sol_ivp.y[0] - S_exact_ivp

# =========================================================================
# SOLUCIÓN USANDO EL MÉTODO MANUAL DE RUNGE-KUTTA DE CUARTO ORDEN (RK4)
# =========================================================================

# Inicializar arreglo para la solución RK4
S_rk4 = np.zeros(len(t_values))
S_rk4[0] = S0 # Establecer la condición inicial

# Implementar el Método de Runge-Kutta de Cuarto Orden (RK4)
for j in range(len(t_values) - 1):
    k1 = F(t_values[j], S_rk4[j])
    k2 = F(t_values[j] + h/2, S_rk4[j] + 0.5 * k1 * h)
    k3 = F(t_values[j] + h/2, S_rk4[j] + 0.5 * k2 * h)
    k4 = F(t_values[j] + h, S_rk4[j] + k3 * h)
    
    # Calcular el siguiente valor de S
    S_rk4[j+1] = S_rk4[j] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

# Calcular la solución exacta y el error para la solución RK4 manual
S_exact_rk4 = exact_solution(t_values)
error_rk4 = S_exact_rk4 - S_rk4 # Error como S_aprox - S_exacta

# =========================================================================
# IMPRESIÓN DE RESULTADOS EN FORMA DE TABLA
# =========================================================================

print("--- Resultados de la aproximación usando scipy.integrate.solve_ivp ---")
df_ivp_resultados = pd.DataFrame({
    't': sol_ivp.t,
    'S_aprox (solve_ivp)': sol_ivp.y[0],
    'S_exacto (sin(t))': S_exact_ivp,
    'Error (solve_ivp)': error_ivp
})
print(df_ivp_resultados.to_string(index=False))
print("-" * 75)
print("\n")

print("--- Resultados de la aproximación usando el Método Manual RK4 ---")
df_rk4_resultados = pd.DataFrame({
    't': t_values,
    'S_aprox (RK4 Manual)': S_rk4,
    'S_exacto (sin(t))': S_exact_rk4,
    'Error (RK4 Manual)': error_rk4
})
print(df_rk4_resultados.to_string(index=False))
print("-" * 75)
print("\n")

# =========================================================================
# GRÁFICOS
# =========================================================================

# --- Gráficos para la solución obtenida con scipy.integrate.solve_ivp ---
# (La gráfica de error aquí es la que se asemeja a tu imagen original)
plt.figure(figsize=(12, 5))

# Subplot 1: Solución aproximada y exacta
plt.subplot(121)
plt.plot(sol_ivp.t, sol_ivp.y[0], label='Aproximación (solve_ivp)')
plt.plot(sol_ivp.t, S_exact_ivp, label='Solución Exacta (sin(t))', linestyle='--')
plt.xlabel("t")
plt.ylabel("S(t)")
plt.title("Solución: solve_ivp vs. Exacta")
plt.legend()
plt.grid(True)

# Subplot 2: Gráfica del error (S_aprox - S_exacta)
plt.subplot(122)
plt.plot(sol_ivp.t, error_ivp)
plt.xlabel("t")
plt.ylabel("Error (S_aprox - S_exacta)")
plt.title("Error de la aproximación (solve_ivp)")
# Ajustar los límites del eje Y para que se parezca a la imagen de referencia
plt.ylim([-0.0012, 0.0016]) 
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Gráficos para la solución obtenida con el método RK4 manual ---
# (Observa que el error es mucho menor y más suave aquí)
plt.figure(figsize=(12, 5))

# Subplot 1: Solución aproximada y exacta
plt.subplot(121)
plt.plot(t_values, S_rk4, label='Aproximación (RK4 Manual)')
plt.plot(t_values, S_exact_rk4, label='Solución Exacta (sin(t))', linestyle='--')
plt.xlabel("t")
plt.ylabel("S(t)")
plt.title("Solución: RK4 Manual vs. Exacta")
plt.legend()
plt.grid(True)

# Subplot 2: Gráfica del error (S_aprox - S_exacta)
plt.subplot(122)
plt.plot(t_values, error_rk4)
plt.xlabel("t")
plt.ylabel("Error (S_aprox - S_exacta)")
plt.title("Error de la aproximación (RK4 Manual)")
plt.grid(True)
plt.tight_layout()
plt.show()