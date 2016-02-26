#-*-coding:utf-8-*-
import functions
import generation
import math
import numpy as np

from Map import Map
from Model import Model
from numpy import unravel_index

def main(size, obstacle_rate, error, path_length, enlarge_factor):
	map_matrix = generation.generate_map(size, obstacle_rate)
	print("Map generated")
	path, observations = generation.generate_sample(map_matrix, path_length)


	map_matrix.compute_a_matrix()
	map_matrix.compute_b_matrix()
	map_matrix.compute_pi_vector()

	final_state = map_matrix.forward(observations)

	best_path = map_matrix.viterbi(observations)

	map_matrix.generate_image(final_state, path, best_path, enlarge_factor)

	forward_error = functions.manhattan_distance(path[len(path) -1], final_state)
	path_error = functions.path_error(path, best_path)

	print("Original state: {0}\nEstimated state: {1}\nError: {2}".format(path[len(path)-1], final_state, forward_error))
	print(path)
	print(observations)
	print("Original path: {0}\nBest path:{1}\nError: {2}".format(path, best_path, path_error))


if __name__ == "__main__":
	main((40, 40), 0.4, 0.01, 15, 50)