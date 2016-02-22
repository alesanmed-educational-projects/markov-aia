#-*-coding:utf-8-*-
import numpy as np

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

	def is_obstacle(self, x, y):
		if self.get_map() == None:
			self.generate_map()

		return bool(self.get_map[x, y])