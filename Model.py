#-*-coding:utf-8-*-
import numpy as np

class Model:
	def __init__(self):
		self.a_matrix = None
		self.b_matrix = None
		self.pi_matrix = None

	def get_a_matrix(self):
		return self.a_matrix

	def set_a_matrix(self, a_matrix):
		self.a_matrix = np.array(a_matrix)

	def get_b_matrix(self):
		return self.b_matrix

	def set_b_matrix(self, b_matrix):
		self.b_matrix = b_matrix

	def get_pi_matrix(self):
		return self.pi_matrix

	def set_pi_matrix(self, pi_matrix):
		self.pi_matrix = pi_matrix

	#Genera la matriz mapa, size es una tupla (filas, columnas) y obstacle_rate es el porcentaje de obstáculos en el mapa
	def compute_a_matrix(map_matrix):
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

	#Calcula el vector pi para el mapa map_matrix
	def compute_pi_matrix(map_matrix):
		pi_matrix = np.zeros(map_matrix.size)
		num_zeros = (map_matrix.size[0]*map_matrix.size[1]) - np.count_nonzero(map_matrix.map_matrix)
		for row in range(map_matrix.size[0]):
			for column in range(map_matrix.size[1]):
				if map_matrix.map_matrix[row][column] == 0:
					pi_matrix[row][column] = 1 / num_zeros
	
	#Calcula la matriz B para el mapa map_matrix y el error error
	def compute_b_matrix(map_matrix, error=0.01):
		return None

	#Private methods
	def get_transition_rate(map_matrix, x, y):
		count = 0.0

		if not map_matrix.is_obstacle(x,y-1):
			count += 1.0
		if not map_matrix.is_obstacle(x,y+1):
			count += 1.0
		if not map_matrix.is_obstacle(x-1,y):
			count += 1.0
		if not map_matrix.is_obstacle(x+1,y):
			count += 1.0