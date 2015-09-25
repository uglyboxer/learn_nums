""" Takes in a vector and set of weights for that vector
Returns their dot-product
"""

def dot_product(vector, weights):
	return sum(elem * weight for elem, weight in zip(vector, weights))

if __name__ == '__main__':
	print(dot_product((1, 1, 1), (.5, .25, .5)))
	print(dot_product((0, 1, 0), (.5, .25, .5)))