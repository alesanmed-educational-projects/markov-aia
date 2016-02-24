#-*-coding:utf-8-*-
from Model import Model
import numpy as np
import functions
import math

class Map(Model):
	def __init__(self, size=(10,10), obstacle_rate=0.4, map_matrix=None, error=0.01):		
		self.size = size
		self.obstacle_rate = obstacle_rate
		self.map_matrix = map_matrix
		self.error = error
		self.a_matrix = None
		self.b_matrix = None
		self.pi_vector = None
		self.state_translation = self.coordinates_to_state #np.arange(size[0]*size[1])

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

	def get_error(self):
		return self.error

	def set_error(self, error):
		self.error = error

	# Estados estan representados de la siguiente manera:
	# 
	#  (Coord)  y->
	#    -------------
	#  x | 0 | 1 | 2 |
	#  | -------------
	#  v | 3 | 4 | 5 |
	#    -------------
	#
	def coordinates_to_state(self, point):
		return point[0] * self.get_size()[1] + point[1]

	# Matriz de transición.
	# 	Probabilidad de pasar de un estado a otro en cualquier momento.
	# 	La entrada A[i][j] es la probabilidad P(x_{t+1} = j | x_t = i) de cambiar de un estado i a j.
	def compute_a_matrix(self):
		shape = (self.get_size()[0]**2, self.get_size()[1]**2)
		a_matrix = np.zeros((shape[0], shape[1]))
		for state1 in range(shape[0]):
			for state2 in range(shape[1]):
				a_matrix[state1, state2] = self.get_transitions_rate(state1, state2)

		self.a_matrix = a_matrix

	# Vector de probabilidades iniciales
	# 	Probabilidad de comenzar en un estado determinado
	# 	La entrada pi[i] es la probabilidad P(x_0 = i) de comenzar en el estado i en el momento 0.
	def compute_pi_vector(self):
		size = self.get_size()
		pi_vector = np.zeros(size[0]*size[1])
		num_zeros = (size[0]*size[1]) - np.count_nonzero(self.map_matrix)
		for row in range(size[0]):
			for column in range(size[1]):
				if self.map_matrix[row][column] == 0:
					pi_vector[self.state_translation((row, column))] = 1 / num_zeros

		self.pi_vector = pi_vector
	
	# Matriz de probabilidad de observación.
	# 	Probabilidad de detectar una observación en un estado concreto
	# 	La entrada B[i][j] es la probabilidad P(y_t = j | x_t = i) de hallar la observación j en el estado i.
	def compute_b_matrix(self):
		shape = self.get_size()
		b_matrix = np.zeros((shape[0]*shape[1], 16))
		for row in range(shape[0]):
			for column in range(shape[1]):
				for obs in range(0,16):
					b_matrix[self.state_translation((row, column))][obs] = self.get_observation_rate(row, column, obs)

		self.b_matrix = b_matrix

	#Genera la matriz mapa
	#   0 -> Hueco libre
	#	1 -> Obstáculo
	# 	Los bordes son siempre obstáculos
	def generate_map(self):
		#Genera una matriz aleatoria de tamaño size, cuyos valores son [0, 1] con la proporcion indicada en p
		map_matrix = np.random.choice([0, 1], size=self.size, p=[1 - self.obstacle_rate, self.obstacle_rate])

		#Se ponen todos los bordes como obstáculos
		map_matrix[0,:] = 1.0
		map_matrix[-1,:] = 1.0
		map_matrix[:,0] = 1.0
		map_matrix[:,-1] = 1.0

		self.map_matrix = map_matrix

	# Devuelve si la posición x,y es un obstáculo
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

	# Devuelve si los dos puntos son adyacentes, basado en la distancia de Manhattan
	# 	(Ver: Functions.manhattan_distance)
	def is_adjacent(self, point1, point2):
		return functions.manhattan_distance(point1, point2) == 1

	# Devuelve la dirección en código numérico entre dos casillas
	# 	0: Norte, 1: Este, 2: Sur, 3: Oeste
	def get_direction(self, departure, arrival):
		direction = -1

		if departure[0] == arrival[0] and departure[1] - arrival[1] == 1:
			direction = 0 #N
		elif departure[1] == arrival[1] and departure[0] - arrival[0] == -1:
			direction = 1 #E
		elif departure[0] == arrival[0] and departure[1] - arrival[1] == -1:
			direction = 2 #S
		elif departure[1] == arrival[1] and departure[0] - arrival[0] == 1:
			direction = 3 #O

		return direction

	# Probabilidad de transición de un estado a otro concreto.
	# 	Se calcula como 1 / numero de posibilidades-no-obstaculos a elegir.
	# 	Si es obstáculo: Probabilidad 0.
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

	# Probabilidad de detectar observacion en un punto de coordenadas concretas
	# 	(x, y) son las coordenadas del punto.
	# 	obs es un número entero (0-15) que representa las observaciones posibles. (Ver: Functions.obscode_to_bitarray)
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

	# Algoritmo forward para modelos ocultos de markov
	# 	Recibe:
	# 		- observations: Lista de observaciones
	# 	Devuelve:
	# 		El estado más probable dada la secuencia
	def forward (self, observations):
		factors, alphas = self.forward_recursive(observations, len(observations)-1, np.empty((0,)))

		de_factor = np.prod(factors)
	
		return alphas/de_factor

	def forward_recursive(self, observations, t, factors):
		states = self.get_size()
		alphas = np.zeros(states)
		if t == 0:
			for row in range(states[0]):
				for column in range(states[1]):
					alphas[row, column] = self.get_b_matrix()[self.state_translation((row, column)), observations[0]]
		else:
			factors, prev_alphas = self.forward_recursive(observations, t-1, factors)
			for row in range(states[0]):
				for column in range(states[1]):
					values = []

					for row_2 in range(states[0]): # TODO: creo que si haces un range(x-1, y+2) range(y-1, y+2) solo hace el for ya por las adyacentes 
						for column_2 in range(states[1]): # asi te ahorras tiempo
							if self.is_adjacent((row_2, column_2), (row, column)):
								direction = self.get_direction((row_2, column_2), (row, column))
								if direction == -1:
									raise ValueError("Direccion = -1, salida {0}, llegada {1}".format((row_2, column_2), (row, column)))

								values.append(self.get_a_matrix()[self.state_translation((row_2, column_2)), self.state_translation((row_2, column_2))]*alphas[row_2, column_2])
							else:
								values.append(0.0)

					values = sum(values)
					alphas[row, column] = self.get_b_matrix()[self.state_translation((row, column)), observations[t]]

		factor = 1 / alphas.sum()
		factors = np.append(factors, factor)
		return factors, alphas*factor

	# Algoritmo de Viterbi para modelos ocultos de Markov.
	# 	Recibe:
	# 		- observations: Lista de observaciones
	# 	Devuelve
	#		La secuencia de estados mas probable para las observaciones recibidas
	def viterbi(self, observations):
		states = self.get_size()
		result = np.zeros((states[0]*states[1], len(observations)))
		backpointer = np.ones((states[0]*states[1], len(observations)), 'int32') * -1

		# initialization
		# 	First column
		# 	Iterate through states
		for i in range(result.shape[0]):
			result[i, 0] = self.get_b_matrix()[i, observations[0]]
			result[i, 0] += self.get_pi_vector()[i]
		
		for t in xrange(1, len(observations)):
			for j in range(result.shape[0]):
				result[j, t] = (result[j, t-1].dot(self.get_b_matrix()[j, observations[t]]) * self.get_a_matrix()).max(0)

				
				result[j, t] *= 

	def viterbi_recursive(observations, t, factors):

		states = self.get_size()
		alphas = np.zeros(states)

		if t == 0:
			for row in range(states[0]):
				for column in range(states[1]):
					alphas[row, column] = math.log10(self.get_b_matrix()[self.state_translation((row, column)), observations[0]]) + math.log10(self.get_pi_vector()[self.state_translation((row, column))])

		else:
			factors, prev_alphas = self.viterbi_recursive(observations, t-1, factors)
			for row in range(states[0]):
				for column in range(states[1]):


					alphas[row, column] = self.get_b_matrix()[self.state_translation((row, column)), observations[t]] * max()

		factor = 1 / alphas.sum()
		factors = np.append(factors, factor)
		return factors, alphas*factor

