import time

import numpy as np


# Definición de la función de Himmelblau
def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Número de muestras aleatorias que se generarán
num_samples = 1000

# Variables para almacenar el mínimo encontrado
min_x, min_y = None, None
min_value = float('inf')

# Iniciar el cronómetro
start_time = time.time()

# Generar puntos aleatorios dentro del rango [-5, 5] para x e y
for _ in range(num_samples):
    x = np.random.uniform(-5, 5)
    y = np.random.uniform(-5, 5)
    value = himmelblau(x, y)
    
    # Actualizar el mínimo si se encuentra uno mejor
    if value < min_value:
        min_value = value
        min_x, min_y = x, y
# Detener el cronómetro
end_time = time.time()

# Mostrar los resultados
print(f"Valores mínimos de (x, y): ({min_x:.5f}, {min_y:.5f})")
print(f"Valor mínimo de la función de Himmelblau: {min_value:.5f}")
print(f"Tiempo de ejecución: {end_time - start_time:.5f} segundos")
