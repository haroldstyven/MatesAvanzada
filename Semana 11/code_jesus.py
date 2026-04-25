import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-poster") 

# 1. Definir la EDO: dS/dt = cos(t)
# Aunque 's' no se usa en este caso particular, la forma estándar es F(t, s)
F = lambda t, s: np.cos(t)

# 2. Definir los parámetros del problema
h = 0.1                            # Tamaño del paso
t = np.arange(0, np.pi, h)         # Vector de tiempo de 0 a pi
S = np.zeros(len(t))               # Arreglo para almacenar la solución aproximada
S[0] = 0                           # Condición inicial: S_0 = 0

# 3. Implementación del Método Runge-Kutta de 4to Orden (RK4)
for j in range(len(t) - 1):
    tj = t[j]
    Sj = S[j]
    
    # Cálculos de k1, k2, k3, k4 según la fórmula
    k1 = F(tj, Sj)
    k2 = F(tj + h/2, Sj + 0.5 * k1 * h)
    k3 = F(tj + h/2, Sj + 0.5 * k2 * h)
    k4 = F(tj + h, Sj + k3 * h)
    
    # Calcular el siguiente valor de S
    S[j+1] = Sj + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

# 4. Solución exacta para el cálculo del error
S_exact = np.sin(t)

# 5. Generar las gráficas
plt.figure(figsize=(12, 4))

# Gráfica izquierda: Solución aproximada
plt.subplot(121)
plt.plot(t, S)
plt.xlabel("t")
plt.ylabel("S(t)")

# Gráfica derecha: Error (Aproximada - Exacta)
plt.subplot(122)
plt.plot(t, S - S_exact)
plt.xlabel("t")
plt.ylabel("S(t) - sin(t)")

plt.tight_layout()
plt.show()