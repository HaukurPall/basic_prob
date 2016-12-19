from math import pow
from enum import Enum
import numpy as np


# We use the logarithm probability implementation provided by teachers

class Features(Enum):
    """An enum to classify different feature types."""
    skip = 0
    continuous = 1
    categorical = 2


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

        # learning rate
        self.alpha = 0.1
        self.SS_tot = 0.0

        self.continuous_feature_count = 0
        self.categorical_features = dict()

        for feature_index, feature in enumerate(feature_selection):
            if feature == Features.categorical:
                self.categorical_features[feature_index] = list()

        self.initialize(data_file_path)

    def get_data_count(self):
        return self.data.shape[0]

    def get_continuous_feature_count(self):
        return self.continuous_feature_count

    def get_categorical_feature_count(self):
        return len(list(self.categorical_features.keys()))

    def initialize(self, data_file_path):
        raw_data = read_data_set(data_file_path)
        # we remove "y" from the raw data
        self.actual_prices, raw_data = remove_prices(raw_data)
        mean_price = sum(self.actual_prices) / len(self.actual_prices)
        for price in self.actual_prices:
            self.SS_tot += pow(price - mean_price, 2)

        raw_data = remove_categorical_features(raw_data, self.categorical_features)
        self.continuous_feature_count = raw_data.shape[1]
        # we set the data with a column full of 1 last.
        self.data = np.append(raw_data, np.ones((len(raw_data), 1)), axis=1)
        for categorical_feature in self.categorical_features.keys():
            # add the dummy variables for the categoricals
            self.data = np.append(self.data, create_dummy_variables(self.categorical_features[categorical_feature]), axis=1)

        # we do feature scaling, we skip the last continuous feature we added as 1 always
        for x in range(self.continuous_feature_count):
            minimum = np.amax(self.data[:, x])
            maximum = np.amin(self.data[:, x])
            feature_mean = np.sum(self.data[:, x]) / self.data.shape[0]
            self.data[:, x] -= feature_mean
            self.data[:, x] *= 1.0 / (maximum - minimum)

        # we cast our data structures to np data structures
        self.weights = np.ones((self.data.shape[1], 1))

    def train(self):
        new_weights = []
        for weight_index in range(self.weights.shape[0]):
            summed_up = 0.0
            for i in range(self.data.shape[0]):
                summed_up += (self.predict(self.data[i]) - self.actual_prices[i]) * self.data[i][weight_index]
            new_weights.append(self.weights[weight_index] - self.alpha * summed_up / self.data.shape[0])
        self.weights = np.array(new_weights)

    def predict(self, features):
        '''

        :param features: array of features
        :param price:
        :return:
        '''
        return np.dot(self.weights.T, features)

    def evaluate_model(self):
        # go over data points and calculate residuals
        # model residuals with mean = 0
        SS_res = 0.0
        for observation in range(self.data.shape[0]):
            prediction = self.predict(self.data[observation])
            SS_res += pow(self.actual_prices[observation] - prediction, 2)
        return 1 - (SS_res / self.SS_tot)

    def get_weights(self):
        return self.weights


def create_dummy_variables(categorical_features):
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
    removed_keys = []
    for key in categorical_variables.keys():
        categorical_variables[key] = raw_data[:, key]
        removed_keys.append(key)
    removed_keys = sorted(removed_keys)
    removed_keys.reverse()
    # we remove keys, the last one first
    for key in removed_keys:
        raw_data = np.delete(raw_data, key, axis=1)
    return raw_data


def remove_prices(raw_data):
    return raw_data[:,  -1:], raw_data[:, :-1]


def read_data_set(data_file_path):
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
    :return:
    """
    while iterations > 0:
        lr_model.train()
        iterations -= 1
        value = lr_model.evaluate_model()
        print(str(value))
    print("Final weights: {}".format(lr_model.get_weights()))


def main():
    lr = LinearRegression([Features.continuous,
                           Features.categorical, #2
                           Features.continuous,
                           Features.continuous,
                           Features.categorical, #5
                           Features.continuous,
                           Features.continuous,
                           Features.continuous,
                           Features.categorical, # 9
                           Features.categorical, # 10
                           Features.categorical, # 11
                           Features.continuous,
                           Features.continuous,
                           Features.continuous],
                          './data/data.txt')
    run_model(lr, 300)


if __name__ == "__main__":
    main()
