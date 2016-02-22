#-*-coding:utf-8-*-
import numpy as np
import functions

class Map:
	def __init__(self):		
		self.size = (10, 10)
		self.obstacle_rate = 0.4
		self.map_matrix = None

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

	def get_transitions_rate(self, x, y):
		possibilities = np.zeros((4,))

		if not self.is_obstacle(x, y):
			if not self.is_obstacle(x,y-1):
				possibilities[0] = 1.0 #Norte
			if not self.is_obstacle(x+1,y):
				possibilities[1] = 1.0 #Este
			if not self.is_obstacle(x,y+1):
				possibilities[2] = 1.0 #Sur
			if not self.is_obstacle(x-1,y):
				possibilities[3] = 1.0 #Oeste
			
			possibilities_size = np.where(possibilities > 0.0)[0].size
			
			#Si podemos ir en alguna direccion
			if possibilities_size:
				#Asignamos a todas las opciones posibles la misma probabilidad
				possibilities = np.divide(possibilities, possibilities_size)

		return possibilities

	def get_observation_rate(self, x, y, obs, error):
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

		res = (error**(4-success)) * ((1-error)**success)
		return res;