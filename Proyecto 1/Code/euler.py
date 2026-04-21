# Librerias
import numpy as np
import matplotlib.pyplot as plt

# Parámetros definidos en el proyecto
m = 1.0    
k = 10.0   
c = 0.8    
x0 = 0.1   
v0 = 0.0   
T = 10     
h = 0.05   # Tamaño de paso

# Preparación de arreglos
N = int(T / h)
t = np.linspace(0, T, N + 1)
x_num = np.zeros(N + 1)
v_num = np.zeros(N + 1)

# Condiciones iniciales
x_num[0] = x0
v_num[0] = v0

# Bucle de iteración: Método de Euler Escalar
for n in range(N):
    x_num[n+1] = x_num[n] + h * v_num[n]
    v_num[n+1] = v_num[n] + h * (-(c/m)*v_num[n] - (k/m)*x_num[n])

# Graficación de resultados
plt.figure(figsize=(10, 6))
plt.plot(t, x_num, 'r--', marker='o', markersize=4, label=f'Euler Escalar (h={h})')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel('Tiempo (t)')
plt.ylabel('Desplazamiento x(t)')
plt.title('Solución de Euler Escalar')
plt.legend()
plt.grid(True)
plt.show()