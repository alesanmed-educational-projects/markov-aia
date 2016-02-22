#-*-coding:utf-8-*-
import numpy as np
import functions

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

	#Genera la matriz mapa, size es una tupla (filas, columnas) y obstacle_rate es el porcentaje de obstáculos en el mapa.
	# Matriz de transición.
	# Probabilidad de pasar de un estado a otro en cualquier momento.
	# La entrada A[i][j][k] es la probabilidad P(x_{t+1} = k | x_t = (i,j)) de cambiar de un estado i a j, tal que k es cada posibilidad de movimiento.
	def compute_a_matrix(self, map_matrix):
		shape = map_matrix.size()
		a_matrix = np.zeros((shape[0], shape[1], 4))
		for row in range(shape[0]):
			for column in range(shape[1]):
				a_matrix[row, column] = map_matrix.get_transitions_rate(column, row)

		self.a_matrix = a_matrix

	#Calcula la matriz pi para el mapa map_matrix.
	# Probabilidad de comenzar en un estado determinado
	# La entrada pi[i][j] es la probabilidad P(x_0 = (i,j)) de comenzar en el estado (i,j) en el momento 0.
	def compute_pi_matrix(map_matrix):
		pi_matrix = np.zeros(map_matrix.size)
		num_zeros = (map_matrix.size[0]*map_matrix.size[1]) - np.count_nonzero(map_matrix.map_matrix)
		for row in range(map_matrix.size[0]):
			for column in range(map_matrix.size[1]):
				if map_matrix.map_matrix[row][column] == 0:
					pi_matrix[row][column] = 1 / num_zeros

		self.pi_matrix = pi_matrix
	
	#Calcula la matriz B para el mapa map_matrix y el error error.
	# Matriz de probabilidad de observación.
	# La entrada B[i][j][k] es la probabilidad P(y_t = k | x_t = (i,j)) de hallar la observación k en el estado (i,j).
	def compute_b_matrix(map_matrix, error=0.01):
		shape = map_matrix.size
		b_matrix = np.zeros((shape[0], shape[1], 4))
		for row in range(shape[0]):
			for column in range(shape[1]):
				for obs in range(0,16):
					b_matrix[row][column][obs] = get_observation_rate(map_matrix, column, row, obs, error)

		self.b_matrix = b_matrix


	def get_observation_rate(map_matrix, x, y, obs, error):
		obs = functions.obscode_to_bitarray(obs)

		n = obs[0]
		e = obs[1]
		s = obs[2]
		w = obs[3]

		aciertos = 0;
		if map_matrix.is_obstacle(x, y-1)==n:
			aciertos++;
		if map_matrix.is_obstacle(x+1, y)==e:
			aciertos++;
		if map_matrix.is_obstacle(x, y+1)==s:
			aciertos++;
		if map_matrix.is_obstacle(x-1, y)==w:
			aciertos++;

		res = (error**(4-aciertos)) * ((1-error)**aciertos)
		return res;