# Librerías
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del sistema
m = 1.0    # Masa
k = 10.0   # Constante del resorte
c = 0.8    # Coeficiente de amortiguamiento
x0 = 0.1   # Posición inicial
v0 = 0.0   # Velocidad inicial
T = 10     # Tiempo total de simulación

# Definición de la solución analítica
def solucion_analitica(t, m, c, k, x0, v0):
    alpha = c / (2 * m)
    omega_d = np.sqrt(4 * m * k - c**2) / (2 * m)
    
    A = x0
    B = (v0 + alpha * x0) / omega_d

    solution = np.exp(-alpha * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
    
    return solution

# Generar datos
t = np.linspace(0, T, 1000)
x_t = solucion_analitica(t, m, c, k, x0, v0)

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(t, x_t, 'b', linewidth=2, label='Solución Analítica')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel('Tiempo (t)')
plt.ylabel('Desplazamiento x(t)')
plt.title('Solución Analítica del Sistema Masa-Resorte Amortiguado')
plt.legend()
plt.grid(True)
plt.show()