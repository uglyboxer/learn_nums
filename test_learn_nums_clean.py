# %load test_foo.py
from nose.tools import assert_equal
from learn_nums_clean import *


class Test_run_blind_data(object):

#THIS IS UNSUITED FOR CURRENT BUILD OF FUNCTION == FAIL IN ORIGINAL DESIGN
    def test_learn_nums(self):
        digits = [[1,0,0], [1,0,1],[1,1,0], [1,1,1]]
        # assert_equal(test_learn_nums(None), None)
        assert_equal(learn_nums(digits, [1,1,1,0]), [.8, -.2, -.1])

        print('Success: test_foo')

def main():
    test = TestFoo()
    test.test_foo()

if __name__ == '__main__':
    main()