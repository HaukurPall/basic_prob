from math import exp, log, sqrt, pi, pow
from enum import Enum
import numpy as np


# We use the logarithm probability implementation provided by teachers

class Features(Enum):
    """An enum to classify different feature types. Use skip=0 to not use a certain feature"""
    skip = 0
    continuous = 1
    categorical = 2
    answer = 3


#
# Feature
#     weight
#     type
#     if categorical then we need to be able to add a new level

class LinearRegression:
    """A Class to represent the Linear Regression model.
    """

    def __init__(self, feature_selection, data_file_path):
        """Constructor

        :param feature_selection: A list which describes what features to use and how.
        If the length of the list does not correspond to the actual feature count of the data then those features will be skipped.
        :param data_file_path: The complete location to the data file
        :raises: ValueError if the length of feature_selection does not
        """
        self.weights = []
        self.feature_selection = feature_selection
        self.data = []
        self.actual_prices = []
        self.SS_tot = 0.0
        # learning rate
        self.alpha = 0.02
        self.categorical_features = dict()
        self.initialize(data_file_path)

    def initialize(self, data_file_path):
        for index, feature in enumerate(self.feature_selection):
            if feature == Features.categorical:
                self.categorical_features[index] = dict()

        raw_data = []
        with open(data_file_path) as data_in:
            for house_data in data_in:
                # we consider all values as float
                features = list(map(float, house_data.split()))
                # we create our x_0 and set it to 1.
                self.actual_prices.append(features.pop(len(features) - 1))
                for index, feature in enumerate(features):
                    if index in self.categorical_features:
                        if feature not in self.categorical_features[index]:
                            self.categorical_features[index][feature] = -1
                features.append(1.0)
                raw_data.append(features)

        number_of_continuous_features = len(raw_data[0])
        number_of_columns = len(raw_data[0])
        for key in self.categorical_features.keys():
            number_of_columns -= 1
            number_of_continuous_features -= 1
            number_of_columns += len(self.categorical_features[key].keys())

        self.data = np.zeros(shape=(len(raw_data), number_of_columns))

        for house_index, house_data in enumerate(raw_data):
            # we skip the categoricals
            index = 0
            for feature_index, feature in enumerate(house_data):
                if feature_index not in self.categorical_features:
                    self.data[house_index][index] = feature
                    index += 1
        # we start from the same location
        for house_index, house_data in enumerate(raw_data):
            # we only take the categoricals
            for feature_index, feature in enumerate(house_data):
                if feature_index in self.categorical_features:
                    # we havn't seen this one before
                    if self.categorical_features[feature_index][feature] == -1:
                        self.categorical_features[feature_index][feature] = index
                        index += 1
                    self.data[house_index][self.categorical_features[feature_index][feature]] = 1

        # we do feature scaling, we skip the last continuous feature we added as 1 always
        for x in range(number_of_continuous_features - 1):
            minimum = np.argmin(self.data[:, x])
            maximum = np.argmax(self.data[:, x])
            feature_mean = np.sum(self.data.T[x]) / self.data.shape[0]
            self.data[:, x] -= feature_mean
            self.data[:, x] *= 1.0 / (maximum - minimum)

        for x in range(self.data.shape[1]):
            self.weights.append(0.0)

        mean_price = sum(self.actual_prices) / len(self.actual_prices)
        for price in self.actual_prices:
            self.SS_tot += pow(price - mean_price, 2)

        self.weights[0] = mean_price
        # we cast our data structures to np data structures
        self.weights = np.array(self.weights)
        self.data = np.array(self.data)

    def train(self):
        new_weights = []
        summed_up = 0.0
        for weight_index in range(self.weights.shape[0]):
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
        for observation in range(len(self.data)):
            prediction = self.predict(self.data[observation])
            SS_res += pow(self.actual_prices[observation] - prediction, 2)
        return 1 - (SS_res / self.SS_tot)
        # what is the variance?
        # R²=1-(sum(x_i-w.T*predict(x_i))²/sum(x_i-(1/n)*sum(x_i))² -> 1


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


def main():
    lr = LinearRegression([Features.continuous,
                           Features.continuous,
                           Features.continuous,
                           Features.categorical,
                           Features.continuous,
                           Features.continuous,
                           Features.continuous,
                           Features.continuous,
                           Features.categorical,
                           Features.continuous,
                           Features.continuous,
                           Features.continuous,
                           Features.continuous,
                           Features.answer],
                          './data/data.txt')
    run_model(lr, 300)


if __name__ == "__main__":
    main()
