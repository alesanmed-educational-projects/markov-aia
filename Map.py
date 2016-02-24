#-*-coding:utf-8-*-
from Model import Model
import numpy as np
import functions

class Map(Model):
	def __init__(self, size=(10,10), obstacle_rate=0.4, map_matrix=None, error=0.01):		
		self.size = size
		self.obstacle_rate = obstacle_rate
		self.map_matrix = map_matrix
		self.error = error
		self.a_matrix = None
		self.b_matrix = None
		self.pi_matrix = None
		self.states_translation = np.arange(size[0]*size[1])

	#Genera la matriz mapa, size es una tupla (filas, columnas) y obstacle_rate es el porcentaje de obstáculos en el mapa.
	# Matriz de transición.
	# Probabilidad de pasar de un estado a otro en cualquier momento.
	# La entrada A[i][j][k] es la probabilidad P(x_{t+1} = k | x_t = (i,j)) de cambiar de un estado i a j, tal que k es cada posibilidad de movimiento.
	def compute_a_matrix(self):
		shape = (self.get_size()[0]**2, self.get_size()[1]**2)
		a_matrix = np.zeros((shape[0], shape[1]))
		for state1 in range(shape[0]):
			for state2 in range(shape[1]):
				a_matrix[state1, state2] = self.get_transitions_rate(state1, state2)

		self.a_matrix = a_matrix

	#Calcula la matriz pi para el mapa map_matrix.
	# Probabilidad de comenzar en un estado determinado
	# La entrada pi[i][j] es la probabilidad P(x_0 = (i,j)) de comenzar en el estado (i,j) en el momento 0.
	def compute_pi_matrix(self):
		size = self.get_size()
		pi_matrix = np.zeros(size[0]*size[1])
		num_zeros = (size[0]*size[1]) - np.count_nonzero(self.map_matrix)
		for row in range(size[0]):
			for column in range(size[1]):
				if self.map_matrix[row][column] == 0:
					pi_matrix[row * size[1] + column] = 1 / num_zeros

		self.pi_matrix = pi_matrix
	
	#Calcula la matriz B para el mapa map_matrix y el error error.
	# Matriz de probabilidad de observación.
	# La entrada B[i][j] es la probabilidad P(y_t = k | x_t = (i,j)) de hallar la observación k en el estado (i,j).
	def compute_b_matrix(self):
		shape = self.get_size()
		b_matrix = np.zeros((shape[0]*shape[1], 16))
		for row in range(shape[0]):
			for column in range(shape[1]):
				for obs in range(0,16):
					b_matrix[row*shape[1] + column][obs] = self.get_observation_rate(row, column, obs)

		self.b_matrix = b_matrix

	def get_size(self):
		return self.size

	def set_size(self, size):
		self.size = size

	def get_obstacle_rate(self):
		return self.obstacle_rate

	def set_obstacle_rate(self, obstacle_rate):
		self.obstacle_rate = obstacle_rate

	def get_map(self):
		return self.map_matrix

	def set_map(self, map_matrix):
		self.map_matrix = np.array(map_matrix)
		self.size = self.map_matrix.shape

	#Genera la matriz mapa, size es una tupla (filas, columnas) y obstacle_rate es el porcentaje de obstáculos en el mapa
	def generate_map(self):
		#Genera una matriz aleatoria de tamaño size, cuyos valores son [0, 1] con la proporcion indicada en p
		map_matrix = np.random.choice([0, 1], size=self.size, p=[1 - self.obstacle_rate, self.obstacle_rate])

		#Se ponen todos los bordes como obstáculos
		map_matrix[0,:] = 1.0
		map_matrix[-1,:] = 1.0
		map_matrix[:,0] = 1.0
		map_matrix[:,-1] = 1.0

		self.map_matrix = map_matrix

	#Devuelve si la posición x,y es un obstáculo
	def is_obstacle(self, x, y):
		if self.map_matrix is None:
			self.generate_map()

		#Si la posición está fuera del tablero, es un obstáculo
		if x < 0 or y < 0:
			res = True
		else:
			try:
				res = bool(self.map_matrix[x, y])
			except IndexError:
				#Si está fuera del tablero, es un obstáculo
				res = True

		return res

	def is_adjacent(self, point1, point2):
		return functions.manhattan_distance(point1, point2) == 1

	def get_transitions_rate(self, state1, state2):
		origin = np.unravel_index(state1, self.get_size())
		goal = np.unravel_index(state2, self.get_size())
		possibilities = np.zeros((4,))
		rate = 0.0

		if (not self.is_obstacle(goal[0], goal[1])) and (not self.is_obstacle(origin[0], origin[1])) and self.is_adjacent(origin, goal):
			if not self.is_obstacle(origin[0],origin[1]-1):
				possibilities[0] = 1.0 #Norte
			if not self.is_obstacle(origin[0]+1,origin[1]):
				possibilities[1] = 1.0 #Este
			if not self.is_obstacle(origin[0],origin[1]+1):
				possibilities[2] = 1.0 #Sur
			if not self.is_obstacle(origin[0]-1,origin[1]):
				possibilities[3] = 1.0 #Oeste
			
			possibilities_size = np.where(possibilities > 0.0)[0].size
			
			#Si podemos ir en alguna direccion
			if possibilities_size:
				#Asignamos a todas las opciones posibles la misma probabilidad
				rate = 1 / possibilities_size

		return rate

	def get_observation_rate(self, x, y, obs):
		obs = functions.obscode_to_bitarray(obs)

		n = obs[0]
		e = obs[1]
		s = obs[2]
		w = obs[3]

		success = 0;
		if self.is_obstacle(x, y-1)==n:
			success += 1
		if self.is_obstacle(x+1, y)==e:
			success += 1
		if self.is_obstacle(x, y+1)==s:
			success += 1
		if self.is_obstacle(x-1, y)==w:
			success += 1

		res = (self.error**(4-success)) * ((1-self.error)**success)
		return res;