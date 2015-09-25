from sklearn import datasets

from check_new import check_new
from learn_loop import learn_loop


""" A rudimentary implementation of a perceptron
Takes in a data set with desired outputs & new unmatched data set
Outputs guesses against new data
"""

def learn_nums(digits, answers):
	""" Takes in a data set and the appropriate answers
	Returns the appropriate weight set
	"""
	weight_set_of = [learn_loop(digits, answers, x) for x in range(10)]	
	return weight_set_of

def run_blind_data(digits, answers, weights):
	""" Brings in a pack of untested data and learned weights
	Returns guesses and success ratio
	"""
	successes = 0
	for temp in range(len(digits)):
		guess = check_new(digits[temp], weights)
		actual = answers[temp]
		print("Computer's guess: {}  Actual #: {}".format(guess, actual))
		if guess == actual:
			successes += 1
	success_ratio = successes/len(digits)
	return successes, success_ratio

if __name__ == '__main__':
	digits = datasets.load_digits()
	answers, answers_to_test = digits.target[:1000], digits.target[1000:]
	sliced_digits, unlearned_digits = digits.data[:1000], digits.data[1000:]
	weights = learn_nums(sliced_digits, answers)
	results = run_blind_data(unlearned_digits, answers_to_test, weights)
	print("Computer was right {} times out of {}".format(results[0], len(digits.data)-1000))
	print("For a correct percentage of {}%".format(results[1]))