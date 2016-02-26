#-*-coding:utf-8-*-
import functions
import numpy as np
import random

from Map import Map

#Genera la matriz mapa, size es una tupla (filas, columnas) y obstacle_rate es el porcentaje de obstáculos en el mapa
def generate_map(size, obstacle_rate = 0.4):
	map_matrix = Map()
	map_matrix.set_size(size)
	map_matrix.set_obstacle_rate(obstacle_rate)
	map_matrix.generate_map()

	return map_matrix

#Generar una muestra de camino por el mapa map_matrix
def generate_sample(map_matrix, steps):
	starting_point = np.where(map_matrix.get_map() == 0)

	index = random.randrange(len(starting_point[0]))
	start_coord = (starting_point[0][index], starting_point[1][index])

	path = [start_coord]
	observations = []
	observation = None
	state_observations = np.empty((16,))
	for obs_code in range(0, 16):
		state_observations[obs_code] = map_matrix.get_observation_rate_coords(path[0][0], path[0][1], obs_code)	

	observation = np.random.choice(range(0, 16), 1, p=state_observations)[0]
	observations.append(observation)

	for i in range(1, steps):
		observation = functions.obscode_to_bitarray(observations[i-1])
		movements = get_movements(map_matrix, path[i-1][0], path[i-1][1], observation)
		
		if np.all(movements == 1):
			break
		else:
			options = np.where(movements == 0)
			index = random.randrange(len(options[0]))
			movement = options[0][index]
			path.append(make_movement(path[i-1][0], path[i-1][1], movement))

		observation = None
		state_observations = np.empty((16,))
		for obs_code in range(0, 16):
			state_observations[obs_code] = map_matrix.get_observation_rate_coords(path[i][0], path[i][1], obs_code)	

		observation = np.random.choice(range(0, 16), 1, p=state_observations)[0]
		observations.append(observation)

	return path, observations

def get_movements(map_matrix, x, y, obs):
	map_element = map_matrix.get_map()
	real_movements = np.array([map_element[x-1,y], map_element[x, y+1], map_element[x+1, y], map_element[x, y-1]])
	obs_movements = np.array(obs)

	return np.logical_or(real_movements, obs_movements) * 1

def make_movement(x, y, movement):
	res = None
	if movement == 0:
		res = (x-1, y) #N
	elif movement == 1:
		res = (x, y+1) #E
	elif movement == 2:
		res = (x+1, y) #S
	elif movement == 3:
		res = (x, y-1) #O

	return res

if __name__ == "__main__":
	map_ = generate_map((10, 10), 0.01)
	print(generate_sample(map_, 0.01, 3))