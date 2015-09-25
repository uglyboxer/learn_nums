from sklearn import datasets

from check_new_iris import check_new_iris
from learn_loop import learn_loop


""" A rudimentary implementation of a perceptron
Takes in a data set with desired outputs & new unmatched data set
Outputs guesses against new data
"""

def learn_nums(digits, answers):
	""" Takes in a data set and the appropriate answers
	Returns the appropriate weight set
	"""
	weight_set_of = [learn_loop(digits, answers, x) for x in range(3)]	
	return weight_set_of

def run_blind_data(digits, answers, weights):
	""" Brings in a pack of untested data and learned weights
	Returns guesses and success ratio
	"""
	successes = 0
	for temp in range(len(digits)):
		guess = check_new_iris(digits[temp], weights)
		actual = answers[temp]
		print("Computer's guess: {}  Actual #: {}".format(guess, actual))
		if guess == actual:
			successes += 1
	success_ratio = successes/len(digits)
	return successes, success_ratio

if __name__ == '__main__':
	digits = datasets.load_iris()
	# answers, answers_to_test = digits.target[:125], digits.target[125:]
	# sliced_digits, unlearned_digits = digits.data[:125], digits.data[125:]

	answers_to_test = [x for x in digits.target[2::5]]
	unlearned_digits = [x for x in digits.data[2::5]]
	answers = digits.target[10:130]
	sliced_digits = digits.data[10:130]


	weights = learn_nums(sliced_digits, answers)
	results = run_blind_data(unlearned_digits, answers_to_test, weights)
	print("Computer was right {} times out of {}".format(results[0], 30))
	print("For a correct percentage of {}%".format(results[1]))