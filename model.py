#-*-coding:utf-8-*-
import numpy as np

#Calcula la matriz pi para el mapa map_matrix
def pi_matrix(map_matrix):
	pi_matrix = np.zeros(map_matrix.size)
	num_zeros = (map_matrix.size[0]*map_matrix.size[1]) - np.count_nonzero(map_matrix.map_matrix)
	for row in range(map_matrix.size[0]):
		for column in range(map_matrix.size[1]):
			if map_matrix.map_matrix[row][column] == 0:
				pi_matrix[row][column] = 1 / num_zeros



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