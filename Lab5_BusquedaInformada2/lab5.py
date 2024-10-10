import numpy as np

# Definición de la función de Himmelblau
def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# Función para generar una nueva solución vecina con límites
def obtener_vecino(x, y, tam_paso=0.5, limite_inferior=-5, limite_superior=5):
    # Genera una dirección aleatoria
    angulo = np.random.uniform(0, 2 * np.pi)
    dx = tam_paso * np.cos(angulo)
    dy = tam_paso * np.sin(angulo)
    
    # Calcula las nuevas coordenadas
    nuevo_x = x + dx
    nuevo_y = y + dy
    
    # Aplica los límites
    nuevo_x = np.clip(nuevo_x, limite_inferior, limite_superior)
    nuevo_y = np.clip(nuevo_y, limite_inferior, limite_superior)
    
    return nuevo_x, nuevo_y

# Función de Recocido Simulado
def recocido_simulado(
    x_inicial, y_inicial,
    temp_inicial=1000,
    temp_minima=1e-8,
    alpha=0.95,
    tam_paso=0.5,
    iteraciones_max=100000,
    limite_inferior=-5,
    limite_superior=5
):
    # Estado actual
    actual_x, actual_y = x_inicial, y_inicial
    energia_actual = himmelblau(actual_x, actual_y)
    
    # Mejor solución encontrada
    mejor_x, mejor_y = actual_x, actual_y
    mejor_energia = energia_actual

    temperatura = temp_inicial
    iteracion = 0

    while temperatura > temp_minima and iteracion < iteraciones_max:
        # Generar una nueva solución vecina
        nuevo_x, nuevo_y = obtener_vecino(actual_x, actual_y, tam_paso, limite_inferior, limite_superior)
        nueva_energia = himmelblau(nuevo_x, nuevo_y)

        # Calcula la diferencia de energía
        delta_E = nueva_energia - energia_actual

        # Decidir si se acepta la nueva solución
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / temperatura):
            actual_x, actual_y = nuevo_x, nuevo_y
            energia_actual = nueva_energia

            # Actualizar el mejor encontrado
            if energia_actual < mejor_energia:
                mejor_x, mejor_y = actual_x, actual_y
                mejor_energia = energia_actual

        # Enfriar la temperatura
        temperatura *= alpha
        iteracion += 1

    return mejor_x, mejor_y, mejor_energia

# Parámetros de inicialización dentro de los límites
x_inicial = np.random.uniform(-5, 5)
y_inicial = np.random.uniform(-5, 5)

# Se ejecuta el algoritmo de Recocido Simulado
mejor_x, mejor_y, mejor_energia = recocido_simulado(
    x_inicial, y_inicial,
    temp_inicial=1000,
    temp_minima=1e-8,
    alpha=0.95,          # Tasa de enfriamiento ajustada para mayor precisión
    tam_paso=0.5,
    iteraciones_max=100000,
    limite_inferior=-5,
    limite_superior=5
)

print(f"Mejor solución encontrada: x = {mejor_x:.6f}, y = {mejor_y:.6f}")
print(f"Valor mínimo de la función: f(x, y) = {mejor_energia:.6f}")
