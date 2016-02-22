#-*-coding:utf-8-*-
import numpy as np

#Genera la matriz mapa, size es una tupla (filas, columnas) y obstacle_rate es el porcentaje de obstáculos en el mapa
def generate_map(size, obstacle_rate = 0.4):
	#Cambiar a matriz aleatoria
	map_matrix = np.zeros(size)

	#Poner a obstáculos todas aquellas casillas cuyo valor sea < obstacle_rate

	#Se ponen todos los bordes como obstáculos
	map_matrix[0,:] = 1.0
	map_matrix[-1,:] = 1.0
	map_matrix[:,0] = 1.0
	map_matrix[:,-1] = 1.0

	return map_matrix

#Generar una muestra de camino por el mapa map_matrix
def generate_sample(map_matrix):
	return None