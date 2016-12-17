from math import exp, log, sqrt, pi
from enum import Enum
# We use the logarithm probability implementation provided by teachers
import project.logarithms

class NormalDistribution(object):
    '''An implementation of the geometric distribution that whose support includes 0.'''

    def __init__(self, median=0.0, variance=0.0):
        '''Constructor

        :param param: The parameter of this geometric distribution
        :raises: ValueError if param is not in [0,1]
        '''

        self.median = median
        self.std_dev = sqrt(variance)
        self.constant = (sqrt(2.0*pi)*self.std_dev)

    def log_prob(self, x):
        '''Compute the log-probability of an observation

        :param x: The observation
        :return: The log-probability of x under this distribution
        :raises: ValueError if x is not a non-negative integer
        '''

        if x % 1 != 0 or x < 0:
            raise ValueError("x is not a non-negative integer")

        return self.success + x * self.failure

    def prob(self, x):
        '''Compute the probability of an observation

        :param x: The observation
        :return: The probability of x under this distribution
        :raises: ValueError if x is not a non-negative integer
        '''
        return exp(0.5*(pow((x-self.median)/self.std_dev, 2)))/self.constant

    def get_param(self):
        '''Get the parameter of this distribution

        :return: The parameter of this distribution
        '''
        return self.median

class Features(Enum):
    '''An enum to classify different feature types. Use skip=0 to not use a certain feature'''
    skip = 0
    continuous = 1
    categorical = 2
#
# Feature
#     weight
#     type
#     if categorical then we need to be able to add a new level

class LinearRegression:
    '''A Class to represent the Linear Regression model.'''

    def __init__(self, feature_selection=[Features.continuous, Features.continuous,Features.continuous],
                 data_file_path='./data/data.txt'):
        '''Constructor

        :param feature_selection: A list which describes what features to use and how.
        If the length of the list does not correspond to the actual feature count of the data then those features will be skipped.
        :param data_file_path: The complete location to the data file
        :raises: ValueError if the length of feature_selection does not
        '''
        self.weights = []
        self.feature_selection = feature_selection
        self.all_features = []
        self.initialize(data_file_path)



    def initialize(self, data_file_path):
        with open(data_file_path) as data:
            for line in data:
                features = line.split()
                self.all_features.append(features)
        # we initialize the weights - we just use the last line.
        for x in range(len(features)):
            self.weights[x] = 1.0

    def train(self):
        # feature * coeff = prediction (continuous features)
        # feature (as categorical) |-> feature_value = ignore default (predictor_0 = 0)
        # feature (as categorical) |-> feature_value = predictor_1 * ceoff_f1_1
        # feature (as categorical) = 1

    def evaluate_model(self, data):
        # go over data points and calculate residuals
        # model residuals with mean = 0
        # what is the variance?
        # R²=1-(sum(x_i-w.T*predict(x_i))²/sum(x_i-(1/n)*sum(x_i))² -> 1

def main():
    read_data()
    train_on_data()
    eval_error() #residuals
    # variance of each normal distribution is expected to be the same but mean varies.
    mean_of_dist_i = np.dot(weigth.T, predictions(x_i))

if __name__ == "__main__":
    main()

mean_of_dist_i = np.dot(weigth.T, predictions(x_i))
