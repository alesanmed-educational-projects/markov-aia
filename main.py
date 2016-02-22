#-*-coding:utf-8-*-
import forward
import generation
import numpy as np
import viterbi

from Map import Map
from Model import Model

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

	observations = generation.generate_sample(map_matrix)

	model = Model()
	model.compute_a_matrix(map_matrix)
	#model.compute_b_matrix(map_matrix)
	model.compute_pi_matrix(map_matrix)

	final_state = forward.run(model, observations)

	best_path = viterbi.run(model, observations)

	print("Final state: {0}".format(final_state))

	print("Best path:\n{0}".format(best_path))


if __name__ == "__main__":
	main((10, 10), 0.4, 0.01)