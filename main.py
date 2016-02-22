#-*-coding:utf-8-*-
import forward
import generation
import model
import numpy as np
import viterbi

def main(size, obstacle_rate, error):
	map_matrix = generation.generate_map(size, obstacle_rate)
	observations = generation.generate_sample(map_matrix)

	pi_vector = model.pi_vector(map_matrix)
	a_matrix = model.a_matrix(map_matrix)
	b_matrix = model.b_matrix(map_matrix, error)

	final_state = forward.run(a_matrix, b_matrix, pi_vector, observations)

	best_path = viterbi.run(a_matrix, b_matrix, pi_vector, observations)

	print("Final state: {0}".format(final_state))

	print("Best path:\n{0}".format(best_path))