from math import pow
from enum import Enum
import sys
import numpy as np
import matplotlib
# this is due to a bug in linux
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt


# We use the logarithm probability implementation provided by teachers

class Features(Enum):
    """An enum to classify different feature types."""
    skip = 0
    continuous = 1
    categorical = 2
    value = 9


class LinearRegression:
    """A Class to represent the Linear Regression model.
    """

    def __init__(self, feature_selection, data_file_path):
        """Constructor

        :param feature_selection: A list which describes what features to use and how.
        :param data_file_path: The complete location to the data file
        """
        self.weights = []
        self.data = np.array([])
        self.actual_prices = []
        self.value_index = []

        # learning rate
        self.alpha = 0.1
        self.SS_tot = 0.0

        self.continuous_feature_count = 0
        self.categorical_features = dict()

        self.skip_features = list()
        for feature_index, feature in enumerate(feature_selection):
            if feature == Features.categorical:
                self.categorical_features[feature_index] = list()
            elif feature == Features.skip:
                self.skip_features.append(feature_index)
            elif feature == Features.value:
                self.value_index = [feature_index]

        self.initialize(data_file_path)

    def initialize(self, data_file_path):
        """Reads a file and prepares the data for learning.

        :param data_file_path: The path to the data file
        """
        raw_data = read_data_set(data_file_path)
        # we remove "y" from the raw data
        raw_data, self.actual_prices = remove_prices(raw_data, self.value_index[0])
        mean_price = sum(self.actual_prices) / len(self.actual_prices)
        for price in self.actual_prices:
            self.SS_tot += pow(price - mean_price, 2)

        # we adjust the indieces of our lists
        adjust_indieces(self.value_index, self.skip_features)
        adjust_indieces(self.value_index, self.categorical_features)
        raw_data, removed_keys = remove_categorical_features(raw_data, self.categorical_features)
        # we need to adjust for the removed categorical features
        adjust_indieces(removed_keys, self.skip_features)

        raw_data = remove_skip_features(raw_data, self.skip_features)
        self.continuous_feature_count = raw_data.shape[1]
        # we set the data with a column full of 1 last.
        self.data = np.append(raw_data, np.ones((len(raw_data), 1)), axis=1)

        for categorical_feature in self.categorical_features.keys():
            # add the dummy variables for the categoricals
            self.data = np.append(self.data, create_dummy_variables(self.categorical_features[categorical_feature]), axis=1)

        # we do feature scaling, we skip the last continuous feature we added as 1 always
        for x in range(self.continuous_feature_count):
            minimum = np.amin(self.data[:, x])
            maximum = np.amax(self.data[:, x])
            feature_mean = np.sum(self.data[:, x]) / self.data.shape[0]
            self.data[:, x] -= feature_mean
            self.data[:, x] *= 1.0 / (maximum - minimum)

        # we cast our data structures to np data structures
        self.weights = np.ones((self.data.shape[1], 1))

    def train(self):
        """ Run one iteration to get new weights
        """
        new_weights = []
        # gradient descent
        for weight_index in range(self.weights.shape[0]):
            summed_up = 0.0
            for i in range(self.data.shape[0]):
                summed_up += ((self.predict(self.data[i]) - self.actual_prices[i]) * self.data[i][weight_index])
            new_weights.append(self.weights[weight_index] - (self.alpha * (summed_up / self.data.shape[0])))
        self.weights = np.array(new_weights)

    def predict(self, features):
        '''Predict based on current weights and given features

        :param features: An array of features
        :return: prediction based on features and weights
        '''
        return np.dot(self.weights.T, features)

    def evaluate_model(self):
        """Evaluate model.
        We use R^2

        :return: An evalutation of the model.
        """
        # go over data points and calculate residuals
        # model residuals with mean = 0
        SS_res = 0.0
        for observation in range(self.data.shape[0]):
            prediction = self.predict(self.data[observation])
            SS_res += pow(self.actual_prices[observation] - prediction, 2)
        return 1 - (SS_res / self.SS_tot)

    def get_weights(self):
        """Return current weights

        :return: Current weights
        """
        return self.weights


def adjust_indieces(removed_keys, list_needs_adjusting):
    """Adjust list index pointers

    :param removed_keys: A list of indexes which have been removed
    :param list_needs_adjusting: A list of indexes which need to be adjusted for elimination
    """
    for key in removed_keys:
        for index, list_index in enumerate(list_needs_adjusting):
            if key < list_index:
                list_needs_adjusting[index] = list_index - 1


def create_dummy_variables(categorical_features):
    """Break a categorical variable to multiple values

    :param categorical_features: An array of assignments the categorical value takes
    :return: an array with a column for each value the categorical features takes
    """
    features = dict()
    index = 0
    # for each categorical feature
    for feature_index in range(categorical_features.shape[0]):
        # we store each value it takes
        if categorical_features[feature_index] not in features:
            # the value the feature takes is the key, value = index
            features[categorical_features[feature_index]] = index
            index += 1
    # array (502-something, 2) f.ex.
    array = np.zeros((categorical_features.shape[0], len(list(features.keys()))))
    for feature_index in range(categorical_features.shape[0]):
        # we only set our key to 1
        array[feature_index][features[categorical_features[feature_index]]] = 1
    return array


def remove_categorical_features(raw_data, categorical_variables):
    """ Remove features from raw data

    :param raw_data: The data
    :param categorical_variables: A dictionary which states what indexes to use for categorical variables
    :return raw data without the categorical variables, a list of removed features
    """
    removed_keys = []
    for key in categorical_variables.keys():
        categorical_variables[key] = raw_data[:, key]
        removed_keys.append(key)
    removed_keys = sorted(removed_keys)
    removed_keys.reverse()
    # we remove keys, the last one first
    for key in removed_keys:
        raw_data = np.delete(raw_data, key, axis=1)
    return raw_data, removed_keys


def remove_skip_features(raw_data, skip_features):
    """ Remove features from raw data

    :param raw_data: The data
    :param skip_features: A list of indexes of features to not use
    :return raw data without features listed
    """
    skip_features.reverse()
    for index in skip_features:
        raw_data = np.delete(raw_data, index, axis=1)
    return raw_data


def remove_prices(raw_data, index):
    """ Remove prices from raw data

    :param index: The index of the value
    :param raw_data: The data
    :return raw data without prices, prices
    """
    values = raw_data[:, index]
    raw_data = np.delete(raw_data, index, axis=1)
    return raw_data, values


def read_data_set(data_file_path):
    """Read file and return a numpy array

    :param data_file_path: File path to data.txt so it can be read
    ":return numpy array of the raw data
    """
    raw_data = []
    with open(data_file_path) as data_in:
        for house_data in data_in:
            # we consider all values as float
            features = list(map(float, house_data.split()))
            raw_data.append(features)
    return np.array(raw_data)


def run_model(lr_model, iterations):
    """A helper funciton to run loops on the linear_regression model

    :param lr_model: The model to train
    :param iterations: The number of iterations to run
    :return: Returns two arrays: iteration, evaluation
    """
    i = 1
    x = list()
    y = list()
    while iterations-i >= 0:
        lr_model.train()
        value = lr_model.evaluate_model()
        i += 1
        x.append(i)
        y.append(value)
        print(str(value))
    print("Final weights: {}".format(lr_model.get_weights()))
    return x, y


def print_continuous_graph(graph, file_path, iterations, number_of_features, ):
    """Create a graph s.t. it tries all features continuous

    :param graph: The graph to use
    :param file_path: To the data
    :param iterations: Number of iterations to do gradient decent
    :param number_of_features: The number of features in the dataset
    """
    for index in range(number_of_features):
        feature_selection = [Features.skip] * number_of_features
        feature_selection[index] = Features.continuous
        feature_selection[13] = Features.value
        lr = LinearRegression(feature_selection, file_path)
        x, y = run_model(lr, iterations)
        label = "{}, cont.".format(str(int(index+1)))
        graph.plot(x, y, label=label)
        graph.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

def main(arguments):
    """Run the linear regression

    :param arguments: Arguments from the command line. 1 is the data path. 2 is the number of iterations.
    """
    number_of_features = 14
    file_path = arguments[1]
    iterations = int(arguments[2])
    graph = plt.subplot(111)
    # print_continuous_graph(graph, file_path, iterations, number_of_features)
    # use a list [Features.continuous, Features.continuous, Features.continuous, ...] to set the feature types to use.
    feature_selection = [Features.skip] * number_of_features
    feature_selection[1] = Features.continuous
    feature_selection[2] = Features.value

    lr = LinearRegression(feature_selection, file_path)
    x, y = run_model(lr, iterations)
    # plotting
    label = "{}, cont.".format("3 (cat), \n6, 11 cont.")
    graph.plot(x, y, label=label)
    graph.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

    plt.ylabel('R²')
    plt.xlabel('Iterations')
    box = graph.get_position()
    graph.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.axis([0, int(arguments[2]), -1.00, 1.00])
    plt.grid(True)
    plt.savefig('out.png')
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
