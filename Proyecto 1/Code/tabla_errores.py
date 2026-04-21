import numpy as np

# 1. Definición de parámetros del sistema 
m = 1.0    # Masa (kg)
k = 10.0   # Constante del resorte (N/m)
c = 0.8    # Coeficiente de amortiguamiento (kg/s)
x0 = 0.1   # Posición inicial (m)
v0 = 0.0   # Velocidad inicial (m/s)
h = 0.1    # Tamaño de paso (s)
T_final = 1.0  # Tiempo final para la tabla

# 2. Preparación de la solución analítica (Caso subamortiguado) [cite: 51, 231]
alpha = c / (2 * m)
omega_d = np.sqrt(4 * m * k - c**2) / (2 * m)
A = x0
B = (v0 + alpha * x0) / omega_d

def posicion_exacta(t):
    # Fórmula: x(t) = e^(-alpha*t) * (A*cos(wd*t) + B*sin(wd*t)) [cite: 57, 231]
    return np.exp(-alpha * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))

# 3. Inicialización de variables para el método numérico
n_pasos = int(T_final / h)
x_n = x0
v_n = v0

# Encabezado de la tabla
print(f"{'Paso (n)':<10} | {'Tiempo (t)':<10} | {'x Analítica':<12} | {'x Euler':<12} | {'Error Absoluto':<15}")
print("-" * 75)

# 4. Ciclo de iteración y cálculo de errores [cite: 77-80, 154-157]
for n in range(n_pasos + 1):
    t_n = n * h
    x_ex = posicion_exacta(t_n)
    error = abs(x_ex - x_n)
    
    # Imprimir fila de la tabla
    print(f"{n:<10} | {t_n:<10.2f} | {x_ex:<12.6f} | {x_n:<12.6f} | {error:<15.6f}")
    
    # Actualización de Euler Escalar para el siguiente paso [cite: 63, 244-245]
    if n < n_pasos:
        # x(n+1) = x(n) + h * v(n)
        x_siguiente = x_n + h * v_n
        # v(n+1) = v(n) + h * (-(c/m)*v(n) - (k/m)*x(n))
        v_siguiente = v_n + h * (-(c/m) * v_n - (k/m) * x_n)
        
        x_n = x_siguiente
        v_n = v_siguiente