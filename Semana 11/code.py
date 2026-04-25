import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Usar un estilo similar al de la imagen (se usa una alternativa moderna para evitar errores)
try:
    plt.style.use("seaborn-v0_8-poster")
except:
    pass # Si la versión es muy antigua o no tiene el estilo, usará el predeterminado

# 1. Definir la función ODE y la solución exacta
def F(t, S):
    return np.cos(t)

def exact_solution(t):
    return np.sin(t)

# 2. Configurar los parámetros iniciales
h = 0.1 # Tamaño de paso inferido del código de la imagen
t = np.arange(0, np.pi, h)
S = np.zeros(len(t))
S[0] = 0 # Condición inicial S_0 = 0

# 3. Implementar el Método de Runge-Kutta de Cuarto Orden (RK4)
for j in range(len(t) - 1):
    k1 = F(t[j], S[j])
    k2 = F(t[j] + h/2, S[j] + 0.5 * k1 * h)
    k3 = F(t[j] + h/2, S[j] + 0.5 * k2 * h)
    k4 = F(t[j] + h, S[j] + k3 * h)
    
    # Calcular el siguiente valor
    S[j+1] = S[j] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

# 4. Calcular la solución exacta y el error
S_exact = exact_solution(t)
error = S - S_exact # Error como está en la gráfica original: S(t) - sin(t)

# ==========================================
# RESULTADOS EN FORMA DE TABLA
# ==========================================
# Creamos un DataFrame con pandas para mostrar la tabla de forma elegante
df_resultados = pd.DataFrame({
    't': t,
    'S_aprox (RK4)': S,
    'S_exacto (sin(t))': S_exact,
    'Error': error
})

print("Resultados de la aproximación mediante Runge-Kutta de Cuarto Orden (RK4):")
print("-" * 75)
print(df_resultados.to_string(index=False)) # Imprime la tabla sin el índice de fila
print("-" * 75)

# ==========================================
# GRÁFICOS
# ==========================================
plt.figure(figsize=(12, 4))

# Subplot 1: Solución aproximada
plt.subplot(121)
plt.plot(t, S)
plt.xlabel("t")
plt.ylabel("S(t)")
plt.title("Aproximación con RK4")

# Subplot 2: Error
plt.subplot(122)
plt.plot(t, error)
plt.xlabel("t")
plt.ylabel("S(t) - sin(t)")
plt.title("Error de la aproximación")

# Ajustar diseño y mostrar
plt.tight_layout()
plt.show()