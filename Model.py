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
	# Algoritmo forward para modelos ocultos de markov
	# 	Recibe:
	# 		- observations: Lista de observaciones
	# 	Devuelve:
	# 		El estado más probable dada la secuencia
	def forward (self, observations):
		alphas = self.forward_recursive(observations, len(observations)-1)
	
		return alphas

	def forward_recursive(self, observations, t):
		states = self.get_b_matrix().shape[0]
		alphas = np.zeros((states,))
		if t == 0:
			for state in range(states):
				alphas[state] = self.get_b_matrix()[state, observations[t]] * self.get_pi_vector()[state]
		else:
			prev_alphas = self.forward_recursive(observations, t-1)
			for state_j in range(states):
				values = 0.0

				for state_i in range(states):
					values += (self.get_a_matrix()[state_i, state_j] * prev_alphas[state_i])
				
				alpha = values * self.get_b_matrix()[state_j, observations[t]]
				alphas[state_j] = alpha

		return alphas / alphas.sum()
		
	"""# Algoritmo de Viterbi para modelos ocultos de Markov.
	# 	Recibe:
	# 		- observations: Lista de observaciones
	# 	Devuelve
	#		La secuencia de estados mas probable para las observaciones recibidas
	def viterbi(self, observations):
		states = self.get_b_matrix().shape[0]
		nus, back_pointers = viterbi_recursive(observations, len(observations) - 1, np.zeros((len(observations) - 1, states)))

		return None

	def viterbi_recursive(observations, t, back_pointers):
		states = self.get_b_matrix().shape[0]
		nus = np.zeros((states,))

		if t == 0:
			for state in range(states):
				nus[state] = self.get_b_matrix()[state, observations[t]] * self.get_pi_vector()[state]
				back_pointers[t, state] = -1

		else:
			prev_nus, back_pointers = self.viterbi_recursive(observations, t-1, back_pointers)

			for state_j in range(states):
				max_value = -1
				for state_i in range(states):
					candidate = 
					values += (self.get_a_matrix()[state_i, state_j] * self.get_pi_vector()[state_i])
				
				alpha = values * self.get_b_matrix()[state_j, observations[t]]
				alphas[state_j] = alpha

		factor = 1 / nus.sum()
		factors = np.append(factors, factor)
		return factors, nus*factor"""