#-*-coding:utf-8-*-
import forward
import functions
import generation
import numpy as np
import viterbi

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

	#path, observations = generation.generate_sample(map_matrix, error, 3)

	map_matrix.compute_a_matrix()
	map_matrix.compute_b_matrix()
	map_matrix.compute_pi_vector()

	print(map_matrix.get_b_matrix())

	"""final_state = forward.run(map_matrix, observations)

	best_path = viterbi.run(map_matrix, observations)

	forward_error = functions.manhattan_distance(path[len(path) -1], unravel_index(np.argmax(final_state), final_state.shape))

	print("Original state: {0}\nEstimated state: {1}\nError: {2}".format(path[len(path)-1], unravel_index(np.argmax(final_state), final_state.shape), forward_error))

	print("Best path:\n{0}".format(best_path))"""


if __name__ == "__main__":
	main((100, 100), 0.1, 0.01)