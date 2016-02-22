#-*-coding:utf-8-*-
import numpy as np

#Calcula el vector pi para el mapa map_matrix
def pi_vector(map_matrix):
	return None

#Calcula la matriz A para el mapa map_matrix
def a_matrix(map_matrix):
	a_matrix = np.zeros(map_matrix.shape)
	for row in range(map_matrix.shape[0]):
		for column in range(map_matrix.shape[1]):
			if row == column:
				a_matrix[row, column] = 0.0 #No se puede permanecer en la misma casilla
			elif map_matrix[row, column]:
				a_matrix[row, column] = 0.0 #No se puede ir a una casilla con obstaculo
			else:
				get_transition_rate(map_matrix, row, column)

	return None

#Calcula la matriz B para el mapa map_matrix y el error error
def b_matrix(map_matrix, error=0.01):
	return None


#Private methods
def get_transition_rate(map_matrix, x, y):
	count = 0.0

	if !map_matrix[x,y-1]:
