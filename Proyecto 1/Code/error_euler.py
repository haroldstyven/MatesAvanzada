import numpy as np
import matplotlib.pyplot as plt

# Parámetros del proyecto
m, k, c = 1.0, 10.0, 0.8
x0, v0, T = 0.1, 0.0, 10

# Solución analítica para comparación
alpha = c / (2 * m)
omega_d = np.sqrt(4 * m * k - c**2) / (2 * m)
A_coef, B_coef = x0, (v0 + alpha * x0) / omega_d

def x_exacta(t):
    return np.exp(-alpha * t) * (A_coef * np.cos(omega_d * t) + B_coef * np.sin(omega_d * t))

# Diferentes tamaños de paso para el análisis
h_values = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
max_errors = []

for h in h_values:
    N = int(T / h)
    t_num = np.linspace(0, T, N + 1)
    x_n, v_n = x0, v0
    max_err = 0
    
    for n in range(N):
        # Euler escalar
        x_next = x_n + h * v_n
        v_next = v_n + h * (-(c/m)*v_n - (k/m)*x_n)
        
        x_n, v_n = x_next, v_next
        
        # Calcular error en este paso
        err_step = abs(x_exacta(t_num[n+1]) - x_n)
        if err_step > max_err:
            max_err = err_step
            
    max_errors.append(max_err)

# Gráfica de convergencia (Log-Log)
plt.figure(figsize=(10, 6))
plt.loglog(h_values, max_errors, 'ro-', linewidth=2, label='Error Máximo (Euler)')
plt.loglog(h_values, [h for h in h_values], 'k--', alpha=0.5, label='Pendiente O(h)') # Referencia
plt.xlabel('Tamaño de paso (h)')
plt.ylabel('Error absoluto máximo')
plt.title('Análisis de Error: Convergencia del Método de Euler')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.show()