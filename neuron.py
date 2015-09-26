# A basic single-neuron Perceptron
# Will learn from one set of data, then make educated guesses as to
# the identity of things described in a second set of data
#
# Title: neuron.py
# Author: Cole Howard 
# Contact: cole@subtlegears.com 
# References:
#
# Usage: python learn_nums.py
#
# Dataset: Written to work on the handwritten digits from scikit-learn
# but datasets can be adjusted in train_and_test() below.
#
# Percptron settings can be adjusted as follows:
#   Learning rate: in Neuron.update_weights() change l_rate
#   Overfitting stop: Change the 'counter' conditional in train_neuron() 

from math import e 

from sklearn import datasets


class Vector:
    """ Model of input set """

    def __init__(self, length):
        """ A list of determined length 

        Args:
            length (int)
        """
        self.length = length


class Neuron:
    """ Model of a neuron for a Perceptron"""

    def __init__(self, vector_length, target = None, sample_size = 1):
        """ __init__ Method

        Args:
            vetor_length (int)
            sample_target (int)
            sample_size (int)

        Generates a list of instances of the Receptor class
        appropriate for length of input vector.

        Generates a list of guesses, one for each vector in the training set.
        """

        self.vector_length = vector_length
        self.target = target
        self.receptors = [0 for x in range(vector_length)]
        self.guess = [0 for x in range(sample_size)]


    def update_weights(self, error, vector):
        """ Updates the weights stored in the receptors

        Args:
            error (int)
            vector(list)
        Returns:
            None
        """
        l_rate = .1
        for idx, item in enumerate(vector):
            self.receptors[idx] += (l_rate * error)  #### Took out -1* just for testing
        return self.receptors


    def fires(self, vector):
        """ Takes an input vector and decides if neuron fires or not 
        
        Args:
            vector - a list 

        Returns:
            a boolean
        """
        sigmoid = activation(vector, self.receptors)
        if sigmoid > .5:
            return True, sigmoid
        else:
            return False, sigmoid


def append_bias(vector):
    """ Takes a list of n entries and appends a 1 for the bias

    Args:
        vector - a list

    Returns:
        a list
    """
    temp_vector = [x for x in vector]
    temp_vector.append(1)
    return temp_vector


def train_neuron(neuron, vectors, answers):
    """ Takes in an instance of a Neuron and a set of training vectors
    
    Args:
        neuron - an instance of a Neuron
        vectors - a list of lists
        answers - a list of ints 

    Returns:
        None
    """  
    expected = [1 if answers[idx] == neuron.target else 0 for idx, x in 
                enumerate(vectors)] 
    counter = 0
    while True:
        for idx, vector in enumerate(vectors):
            fired, sigmoid = neuron.fires(vector)    # Activation resul
            if fired:  
                neuron.guess[idx] = 1
            else:
                neuron.guess[idx] = 0
        ### Here is where you calculate the error to feed into weights
        ### y is the activation
        ### target is 1 or 0
        ### weight += -1(l_rate)(gradient)
        ### Then let T be the vector of target values, Y be the current output
        ### vector, and msq be the error function. You will need to 
        ### backpropagate the derivative d/dy msq(T, Y), for each y in Y
        ### which should == y(1-y) y = sigmoid (s)
            err = expected[idx] - neuron.guess[idx]
            error = sigmoid * (sigmoid-1) * err
            print(error) 
            neuron.update_weights(error, vector)
        if expected != neuron.guess and counter < 500: #To prevent overfitting
            counter += 1
            continue
        else:
            break
    return None


def activation(vector, weights):
    """ Activation function on two equal length vectors

    Args:
        vector (list)
        weights(list)

    Returns:
        a float
    """

    s = sum(elem * weight for elem, weight in zip(vector, weights))
    return (1/(1 + e**(-s)))


def computer_guess(vector, neurons):
    """Takes in a vector and passes it through each neuron
    and appends that neuron's index if it fires.
    
    Args:
        vector - a list
        neurons - a list of Neuron instances

    Returns:
        guess - an int
        or, None
    """
    guesses = [(activation(vector, neurons[x].receptors), x) 
        for x in range(len(neurons)) if neurons[x].fires(vector)]
    guesses.sort(reverse=True)
    try:
        return guesses[0][1]
    except:
        return None


def train_and_test():
    """ This takes the dataset and breaks it into a traing set and a test set.
    It assigns a neuron to each possible outcome and trains each neuron on
    the entire training set.
        Finally it passes the test set through each neuron and guesses the
    answer based on which fires.

    DATA SETS:
        This runs on the scikit-learn handwritten numbers dataset, but that
        can be swapped out in the first 4 lines below.

    Outputs:
        Returns the correct number of guesses against the total in the test
        set and the success rate.
    """

    # Dependent on input set
    digits = datasets.load_digits()
    target_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    answers, answers_to_test = digits.target[:1000], digits.target[1000:]
    training_set, testing_set = digits.data[:1000], digits.data[1000:]

    # For all inputs
    vector_length = len(training_set[0])
    neurons = [Neuron(vector_length + 1, x, len(training_set)) for x in 
                target_values]
    vectors = [append_bias(vector) for vector in training_set]
    test_vectors = [append_bias(vector) for vector in testing_set]
    
    for x in target_values:
        train_neuron(neurons[x], vectors, answers)
    
    successes = 0
    for idx, y in enumerate(testing_set):
        # get computer guess
        g = computer_guess(y, neurons)
        if g == answers_to_test[idx]:
            successes += 1
    print('Computer got {} right out of {} samples.'.format(successes, 
            len(testing_set)))
    print('For a success rate of {}'.format(successes/len(testing_set)))
       

if __name__ == '__main__':
    assert append_bias([3, 4, 5, 6]) == [3, 4, 5, 6, 1]
    assert append_bias([]) == [1]
    # assert activation([4, 3, 2], [2, 3, 4]) == 25
    dummy = Neuron(2,2,2)
    # assert dummy.fires([6, 3]) == False
    # assert [int(x) for x in dummy.update_weights(100, [6, 3])] == [30, 15]
    print("Tests pass.  Initiate happy dance.")


    ### Finish these tests, somehow
    #assert train_neuron(????) == ????
    train_and_test()

__author__ = 'Cole Howard'

