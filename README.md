# Matemáticas Avanzadas - Maestría en Ingeniería

Repositorio académico que contiene los materiales, ejercicios, tareas y proyectos del curso de **Matemáticas Avanzadas** del primer semestre de la Maestría en Ingeniería.

## 📋 Contenido General

Este curso cubre los fundamentos de ecuaciones diferenciales ordinarias (EDO), métodos numéricos para su resolución y aplicaciones prácticas en ingeniería.

---

## 📚 Estructura del Curso

### **Semana 1: Introducción a Ecuaciones Diferenciales**

Fundamentos básicos de las ecuaciones diferenciales ordinarias y métodos de resolución.

**Temas cubiertos:**
- **Ecuaciones Separables**: Técnica de separación de variables para resolver EDO de primer orden
  - Documentos: Problemas de Zill y Kreyszig sobre ecuaciones separables
  - Materiales de referencia en PDF

- **Fórmulas y Definiciones**: Conceptos fundamentales y terminología
  - Definiciones y terminología de EDO
  - Fórmulas de diferenciación e integración

---

### **Semana 2: Series de Taylor**

Desarrollo de funciones mediante series de Taylor y sus aplicaciones en aproximación numérica.

**Temas cubiertos:**
- Expansión en serie de Taylor
- Problemas resueltos (Problemas 2 y 5)
- Aplicaciones en aproximación de soluciones

---

### **Semana 3: Ecuaciones Exactas y Métodos de Diferencias Finitas**

Técnicas avanzadas para resolver ecuaciones diferenciales exactas y métodos numéricos.

**Temas cubiertos:**
- **Ecuaciones Exactas**: 
  - Factores integrantes
  - Problemas de Kreyszig
  - Referencia Zill
  
- **Aproximación por Diferencias Finitas**:
  - Métodos numéricos para aproximar derivadas
  - Problemas computacionales resueltos
  - Visualizaciones gráficas de resultados

---

### **Semana 4: Evaluación**

**Evaluación del Primer Parcial**
- Examen resuelto manualmente
- Problemas integradores de los temas de Semanas 1-3

---

### **Semana 5: Ecuaciones Diferenciales Lineales**

Métodos de resolución para ecuaciones diferenciales lineales y casos especiales.

**Temas cubiertos:**
- Ecuaciones diferenciales lineales de primer orden
- Ecuaciones de Bernoulli
- Ecuaciones de Riccati
- Problemas resueltos de Kreyszig
- Guías de preparación para examen

---

### **Semana 6: Ecuaciones de Bernoulli y Riccati**

Profundización en ecuaciones especiales y técnicas de sustitución.

**Temas cubiertos:**
- Ecuaciones de Bernoulli
- Ecuaciones de Riccati
- Tareas y ejercicios prácticos
- Materiales de referencia de Zill

---

### **Semana 9: Ecuaciones Diferenciales de Segundo Orden**

Resolución de ecuaciones diferenciales de orden superior y sistemas.

**Temas cubiertos:**
- Ecuaciones diferenciales de segundo orden no homogéneas
- Ecuaciones con coeficientes constantes
- Métodos de solución general
- Tests evaluativos
- Materiales de referencia completos

---

### **Semana 10: Aplicaciones Prácticas**

Implementación y resolución de problemas aplicados usando métodos numéricos.

**Temas cubiertos:**
- Problem 2: Aplicaciones de métodos numéricos
- Problem 3: Casos de estudio avanzados
- Documentos de tareas (HW4_AdvMath.pdf)
- Visualizaciones gráficas de soluciones

---

## 🚀 Proyecto 1: Método de Euler para Sistemas de Ecuaciones Diferenciales

### Descripción

Implementación numérica del **método de Euler** para resolver sistemas de ecuaciones diferenciales ordinarias. Aplicación a un sistema dinámico de masa-resorte-amortiguador.

### Contenido del Proyecto

**Códigos Python (`Code/`):**

- **`euler.py`**: Implementación del método de Euler escalar
  - Resolución numérica de sistemas de EDO
  - Simulación de movimiento oscilatorio amortiguado
  - Parámetros del sistema: masa (m), rigidez del resorte (k), amortiguación (c)
  - Condiciones iniciales y discretización temporal

- **`analitica.py`**: Solución analítica del sistema
  - Cálculo de la solución exacta para comparación
  - Funciones de posición y velocidad exactas

- **`error_euler.py`**: Análisis de errores del método numérico
  - Estimación del error de truncamiento
  - Comparación de métodos de diferentes órdenes

- **`tabla_errores.py`**: Generación de tablas de comparación
  - Error absoluto y relativo
  - Análisis de convergencia del método
  - Diferentes tamaños de paso

**Documentos (`Documents/`):**
- Especificaciones del proyecto
- Enunciados y objetivos

**Gráficos (`graphs/`):**
- Soluciones numéricas vs analíticas
- Análisis de errores
- Visualización del comportamiento del sistema

**Imágenes (`Img/`):**
- Gráficas de resultados comparativos
- Análisis de errores de truncamiento
- Visualización de movimiento de proyectiles
- Comparaciones de soluciones

### Objetivos

1. Implementar el método de Euler para sistemas de EDO
2. Validar resultados numéricos contra soluciones analíticas
3. Analizar el error de truncamiento local y global
4. Estudiar la convergencia del método

### Resultados Clave

- Visualización de soluciones numéricas vs analíticas
- Análisis detallado de errores de truncamiento
- Comparativas de diferentes tamaños de paso
- Comportamiento del sistema dinámico bajo diferentes parámetros

---

## 📂 Estructura de Directorios

```
Matematicas Avanzadas/
├── Proyecto 1/
│   ├── Code/                  # Código Python para el proyecto
│   │   ├── analitica.py
│   │   ├── error_euler.py
│   │   ├── euler.py
│   │   └── tabla_errores.py
│   ├── Documents/             # Especificaciones y enunciados
│   ├── Img/                   # Imágenes y gráficos generados
│   ├── Slides/                # Presentaciones
│   └── graphs/                # Gráficos de resultados
├── Semana 1/
│   ├── Ecuaciones Separables/ # Materiales sobre ecuaciones separables
│   └── Formulas/              # Fórmulas básicas y definiciones
├── Semana 2/
│   └── Serie de Taylor/       # Expansión en series de Taylor
├── Semana 3/
│   ├── Ecuaciones Exactas/    # Métodos para ecuaciones exactas
│   └── Finite Difference Approximation/  # Métodos numéricos
├── Semana 4/                  # Evaluación del primer parcial
├── Semana 5/                  # Ecuaciones diferenciales lineales
├── Semana 6/                  # Ecuaciones de Bernoulli y Riccati
├── Semana 9/                  # EDOs de segundo orden
├── Semana 10/                 # Aplicaciones prácticas
└── README.md                  # Este archivo
```

---

## 🎯 Objetivos del Curso

- Dominar la teoría de ecuaciones diferenciales ordinarias
- Implementar y aplicar métodos numéricos para EDO
- Analizar errores y convergencia de métodos numéricos
- Resolver problemas prácticos de ingeniería usando EDO
- Desarrollar habilidades computacionales en Python

---

## 📖 Referencias Bibliográficas

- **Zill, Dennis G.**: "Ecuaciones Diferenciales con Problemas de Valores en la Frontera"
- **Kreyszig, Erwin**: "Advanced Engineering Mathematics"
- Materiales adicionales de clase y problemas resueltos

---

## 🔧 Herramientas Utilizadas

- **Python 3.x**: Implementación de métodos numéricos
- **NumPy**: Cálculos numéricos
- **Matplotlib**: Visualización de resultados
- **LaTeX**: Documentación matemática

---

## ✍️ Autor

Harold Styven Lagares De Voz
Estudiante de Maestría en Ingeniería  
Curso: Matemáticas Avanzadas  
Semestre: 1er semestre

---

## 📝 Notas Finales

Este repositorio contiene todos los materiales necesarios para el seguimiento del curso de Matemáticas Avanzadas, incluyendo teoría, práctica computacional y evaluaciones. Se recomienda revisar los materiales en orden secuencial por semana para una mejor comprensión de los temas.