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