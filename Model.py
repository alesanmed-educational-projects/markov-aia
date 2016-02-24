#-*-coding:utf-8-*-
import numpy as np

class Model:
	def __init__(self, a_matrix, b_matrix, pi_vector, state_translation):
		self.a_matrix = a_matrix
		self.b_matrix = b_matrix
		self.pi_vector = pi_vector
		self.state_translation = state_translation

	def get_a_matrix(self):
		return self.a_matrix

	def set_a_matrix(self, a_matrix):
		self.a_matrix = np.array(a_matrix)

	def get_b_matrix(self):
		return self.b_matrix

	def set_b_matrix(self, b_matrix):
		self.b_matrix = b_matrix

	def get_pi_vector(self):
		return self.pi_vector

	def set_pi_vector(self, pi_vector):
		self.pi_vector = pi_vector

	def compute_a_matrix(self):
		raise NotImplementedError

	def compute_pi_vector(self):
		raise NotImplementedError
	
	def compute_b_matrix(self):
		raise NotImplementedError

	# Algoritmo de Viterbi para modelos ocultos de Markov.
	# 	Recibe:
	# 		- observations: Lista de observaciones
	# 	Devuelve
	#		La secuencia de estados mas probable para las observaciones recibidas
	def viterbi(self, observations):
		n_states = self.get_a_matrix().shape[0]
		result = np.zeros((n_states, len(observations)))
		backpointer = {}

		# initialization
		# 	First column
		# 	Iterate through states
		for i in range(n_states):
			result[i, 0] = self.get_b_matrix()[i, observations[0]]
			result[i, 0] += self.get_pi_vector()[i]
			backpointer[i] = None
		
		for t in range(1, len(observations)):
			#print(str(t))
			for j in range(n_states):

				result[j,t] = (self.get_a_matrix()[j, :] * result[j,t-1]).max(0) * self.get_b_matrix()[j, observations[t]]

				backpointer[j] = (self.get_a_matrix()[j, :] * result[j,t-1]).argmax(0)
				#print("Estado " + str(j) + " + Observación " + str(t) +" -> " + str(backpointer[j, t]))

		s = (result[:, len(observations)-1]).argmax(0)
		if s in backpointer:
			print(backpointer)

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