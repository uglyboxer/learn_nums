from dot_product import dot_product
from update_weights import update_weights
from convert_dataset import convert_dataset


def learn_loop(dataset, answers, target):		### Just the digits from main dataset
	"""Takes a dataset with expected values
	returns proper weights for each element of the vector space
	"""

	# set initial guess, initial weights, threshold, and learning rate
	binary_dataset = convert_dataset(dataset)
	[vector.append(1) for vector in binary_dataset]
	expected = [1 if answers[idx] == target else 0 for idx, x in enumerate(binary_dataset)] 
	weights = [0 for elem in binary_dataset[0]]
	guesses = [0 for vector in binary_dataset]
	threshold = .5
	l_rate = .05 

	### Check each vector against expect and loop over again until 0 errors
	counter = 0
	while True:
		for idy, vector in enumerate(binary_dataset):
			error = expected[idy] - guesses[idy]
			if dot_product(vector, weights) > threshold:   	#if true, set output to 1
				guesses[idy] = 1 
			else:
				guesses[idy] = 0
			weights = update_weights(weights, error, l_rate, vector)
		if expected != guesses and counter < 500:
			counter += 1
			continue
		else:
			break
	return weights

if __name__ == '__main__':
	print("Test this")