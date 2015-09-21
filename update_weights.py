def update_weights(weights, error, l_rate, vector):
	"""Takes in a given weight, error, learning rate, and vector element
	Retuns updated weight
	"""
	return [weight + (elem * l_rate * error) for elem, weight in zip(vector, weights)]

if __name__ == '__main__':
	print(update_weights((.1, .1, .2), -6, .1, (.5, .5, .5)))
