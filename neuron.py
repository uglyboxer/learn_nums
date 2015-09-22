class Receptor:
	""" Model of photo-receptor for a Perceptron """

	### Method for updating weights

class Neuron:
	""" Model of a neuron for a Perceptron"""

	def __init__(self, sample_set, sample_target):
		""" __init__ Method

		Args:
			sample_set (list(list)): List of input vectors(each a list)
			sample_target (list(int)): List of target ints

		"""

		self.sample_set = sample_set
		self.sample_target = sample_target

		### add method for creating list of receptors (list comprehension)