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
	def compute_a_matrix(self, map_matrix):
		shape = map_matrix.get_map().shape
		a_matrix = np.zeros((shape[0], shape[1], 4))
		for row in range(shape[0]):
			for column in range(shape[1]):
				a_matrix[row, column] = map_matrix.get_transitions_rate(row, column)

		self.a_matrix = a_matrix

	#Calcula el vector pi para el mapa map_matrix
	def compute_pi_matrix(self, map_matrix):
		pi_matrix = np.zeros(map_matrix.size)
		num_zeros = (map_matrix.size[0]*map_matrix.size[1]) - np.count_nonzero(map_matrix.map_matrix)
		for row in range(map_matrix.size[0]):
			for column in range(map_matrix.size[1]):
				if map_matrix.map_matrix[row][column] == 0:
					pi_matrix[row][column] = 1 / num_zeros

		self.pi_matrix = pi_matrix
	
	#Calcula la matriz B para el mapa map_matrix y el error error
	def compute_b_matrix(map_matrix, error=0.01):
		return None

	#Private methods
