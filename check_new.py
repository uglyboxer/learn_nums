from dot_product import dot_product

def check_new(vector, weights):
	""" Takes in a vector (representing a number) and weights
	Returns a guess as so it's value
	"""
	threshold = .5
	computer_guesses = []
	for x in range(10):
		if dot_product(vector, weights[x]) > threshold:
			computer_guesses.append(x)
	return computer_guesses
