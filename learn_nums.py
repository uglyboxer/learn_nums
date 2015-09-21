from sklearn import datasets

from check_new import check_new
from learn_loop import learn_loop


""" A rudimentary implementation of a perceptron
Takes in a data set with desired outputs & new unmatched data set
Outputs guesses against new data
"""

digits = datasets.load_digits()
answers = digits.target
sliced_digits = digits.data[:1000]
weight_set_of = [learn_loop(sliced_digits, answers, x) for x in range(10)]		#gets back weights
successes = 0
for temp in range(1001, len(digits.data)):
	guess = check_new(digits.data[temp], weight_set_of)
	actual = digits.target[temp]
	print("Computer's guess: {}  Actual #: {}".format(guess, actual))
	if guess == actual:
		successes += 1
print("Computer was right {} times out of {}".format(successes, len(digits.data)-1000))
print("For a correct percentage of {}%".format(successes/(len(digits.data)-1000)))

