#-*-coding:utf-8-*-
import functions
import generation
import numpy as np

from Map import Map
from Model import Model
from numpy import unravel_index

def main(size, obstacle_rate, error):
	#map_matrix = generation.generate_map(size, obstacle_rate)
	####DEV####
	map_matrix = Map()
	map_matrix.set_map([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
						[1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
						[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
						[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
						[1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
						[1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
						[1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
						[1, 0, 0, 1, 1, 1, 0, 0, 0, 1],
						[1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
						[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	path, observations = generation.generate_sample(map_matrix, 3)

	map_matrix.compute_a_matrix()
	map_matrix.compute_b_matrix()
	map_matrix.compute_pi_vector()

	#print(map_matrix.get_b_matrix())

	final_state = map_matrix.forward(observations)

	#best_path = map_matrix.viterbi(observations)

	forward_error = functions.manhattan_distance(path[len(path) -1], final_state)

	print("Original state: {0}\nEstimated state: {1}\nError: {2}".format(path[len(path)-1], final_state, forward_error))

	#print("Best path:\n{0}".format(best_path))


if __name__ == "__main__":
	main((100, 100), 0.1, 0.01)