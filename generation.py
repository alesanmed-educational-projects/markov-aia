#-*-coding:utf-8-*-
from Map import Map
import numpy as np

#Genera la matriz mapa, size es una tupla (filas, columnas) y obstacle_rate es el porcentaje de obstáculos en el mapa
def generate_map(size, obstacle_rate = 0.4):
	map_matrix = Map()
	map_matrix.set_size(size)
	map_matrix.set_obstacle_rate(obstacle_rate)
	map_matrix.generate_map()

	return map_matrix

#Generar una muestra de camino por el mapa map_matrix
def generate_sample(map_matrix):
	return None

if __name__ == "__main__":
	print(generate_map((10, 10), 0.1).get_map())