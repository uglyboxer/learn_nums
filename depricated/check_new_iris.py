from dot_product import dot_product

def check_new_iris(vector, weights):
	""" Takes in a vector (representing a picture of a number) and weights
	Returns a guess as to it's value
	"""
	threshold = .5
	computer_guesses = []
	dot_product_list = []
	for x in range(3):
		dot_product_list.append(dot_product(vector, weights[x]))
		if dot_product_list[x] > threshold:
			computer_guesses.append(x)
	if dot_product_list:
		return dot_product_list.index(max(dot_product_list))
	else:
		return None

