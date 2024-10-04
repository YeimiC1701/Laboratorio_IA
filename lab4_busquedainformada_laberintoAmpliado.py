import heapq
import time

import numpy as np

# Se define el laberinto como una lista de listas
laberinto = [
    [1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0],
    [1, 0, 1, 1, 1],
    [1, 0, 0, 1, 1],
    [1, 1, 0, 0, 1],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]
]

# En una lista de tuplas se definen las direcciones de movimiento (abajo, arriba, derecha, izquierda) que se pueden hacer dentro del laberinto
direcciones = [(1, 0), (-1, 0), (0, 1), (0, -1)]

#A continuación, se define la heurística que calcula la distancia de Manhattan entre dos puntos
def heuristica(nodo, objetivo):
    return abs(nodo[0] - objetivo[0]) + abs(nodo[1] - objetivo[1])

#Implementación del algoritmo A* para encontrar el camino en el laberinto
def algoritmo_a(laberinto, nodo_inicial, objetivo):
    nodos_frontera = []
    heapq.heappush(nodos_frontera, (0, nodo_inicial))  # (costo total, posición)

    nodos_visitados = {}
    costo_actual = {nodo_inicial: 0}
    puntaje_total = {nodo_inicial: heuristica(nodo_inicial, objetivo)}

    while nodos_frontera:
        nodo_actual = heapq.heappop(nodos_frontera)[1]

        # Si hemos llegado al objetivo, reconstruimos el camino
        if nodo_actual == objetivo:
            camino = []
            while nodo_actual in nodos_visitados:
                camino.append(nodo_actual)
                nodo_actual = nodos_visitados[nodo_actual]
            return camino[::-1]  # Retorna el camino invertido

        for direccion in direcciones:
            nodo_hijo = (nodo_actual[0] + direccion[0], nodo_actual[1] + direccion[1])

            # Comprobamos si el vecino está dentro de los límites del laberinto
            if 0 <= nodo_hijo[0] < len(laberinto) and 0 <= nodo_hijo[1] < len(laberinto[0]):
                if laberinto[nodo_hijo[0]][nodo_hijo[1]] == 0:  # Solo consideramos caminos transitables
                    nuevo_costo = costo_actual[nodo_actual] + 1

                    if nuevo_costo < costo_actual.get(nodo_hijo, float('inf')):
                        nodos_visitados[nodo_hijo] = nodo_actual
                        costo_actual[nodo_hijo] = nuevo_costo
                        puntaje_total[nodo_hijo] = nuevo_costo + heuristica(nodo_hijo, objetivo)

                        # Si el vecino no está en nodos_frontera, lo añadimos
                        if nodo_hijo not in [i[1] for i in nodos_frontera]:
                            heapq.heappush(nodos_frontera, (puntaje_total[nodo_hijo], nodo_hijo))

    return None  # Retorna None si no hay solución

# Definimos los nodos de inicio y fin
nodo_inicial = (0, 1)  # Nodo A
objetivo = (13, 0)   # Nodo I

# Se ejecuta el algoritmo
tiempo_inicio = time.time()
camino = algoritmo_a(laberinto, nodo_inicial, objetivo)
tiempo_final = time.time()
tiempo_ejecucion = str(round(tiempo_final - tiempo_inicio,7))


# Mostramos el resultado
if camino:
    print("Tiempo de ejecución: "+ tiempo_ejecucion + " segundos")
    print("El camino encontrado es:")
    for paso in camino:
        print(paso)
else:
    print("No se encontró un camino.")