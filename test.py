#-*-coding:utf-8-*-
import functions
import generation
import math
import numpy as np
import time

from Map import Map
from Model import Model
from numpy import unravel_index

def test(size, obstacle_rate, error, path_length):
	map_matrix = generation.generate_map(size, obstacle_rate)

	map_matrix.compute_a_matrix()
	map_matrix.compute_b_matrix()
	map_matrix.compute_pi_vector()

	current_milli_time = lambda: int(round(time.time() * 1000))

	forward_errors = []
	path_errors = []
	times = []
	for i in range(100):
		print("Iteracion {0}/{1}".format(i, 99))
		path, observations = generation.generate_sample(map_matrix, path_length)

		t = current_milli_time()
		final_state = map_matrix.forward(observations)

		best_path = map_matrix.viterbi(observations)

		forward_error = functions.manhattan_distance(path[len(path) -1], final_state)
		path_error = functions.path_error(path, best_path)

		t = current_milli_time() - t
		times.append(t)
		forward_errors.append(forward_error)
		path_errors.append(path_error)


	print("Average forward error: {0}\nAverage path error: {1}\nAverate time: {2}".format(np.average(forward_errors), np.average(path_errors), np.average(times)))


if __name__ == "__main__":
	test((50, 50), 0.4, 0.01, 10)