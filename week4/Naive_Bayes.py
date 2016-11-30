from collections import Counter
from functools import reduce
import math


class Naive_Bayes(object):
    '''
    This class implements a naive bayes classifier.
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.label_probs = Counter()
        # A dictionary that maps labels to Counters. Each of these Counters
        # contains the feature probabilities given the label.
        self.feature_probs = dict()
        self.vocabulary = set()

        self.label_log_probs = dict()
        self.label_feature_log_probs = dict()
        self.delta = 1.5

    def train(self, data, label):
        '''
        Train the classifier by counting features in the data set.
        
        :param data: A stream of string data from which to extract features
        :param label: The label of the data
        '''
        for line in data:
            self.add_feature_counts(line.lower().split(), label)

    def add_feature_counts(self, features, label):
        '''
        Count the features in a feature list.
        
        :param features: a list of features.
        :param label: the label of the data file from which the features were extracted.
        '''
        for feature in features:
            self.vocabulary.add(feature)
            if label not in self.feature_probs:
                self.feature_probs[label] = Counter()
            self.feature_probs[label].update([feature])

    def smooth_feature_counts(self, smoothing=1):
        '''Smooth the collected feature counts

        :param smoothing: The smoothing constant
        '''
        for feature in self.vocabulary:
            for label in self.feature_probs.keys():
                if feature not in self.feature_probs[label]:
                    self.feature_probs[label][feature] = smoothing
                else:
                    self.feature_probs[label][feature] += smoothing

    def update_label_count(self, label):
        '''
        Increase the count for the supplied label by 1.
        
        :param label: The label whose count is to be increased.
        '''
        self.label_probs.update([label])

    def log_normalise_label_probs(self):
        '''
        Normalize the label counts to probabilities and transform them to logprobs.
        '''
        # Hint: you can assign a new value to label to label_probs or do the normalisation in place
        label_list = self.label_probs.most_common()
        total_labels = reduce(lambda current, count: current+count[1], label_list, 0)
        for (x, y) in label_list:
            self.label_log_probs[x] = math.log2(float(y)/total_labels)

    def log_normalise_feature_probs(self):
        '''
        Normalize the feature counts for each label to probabilities and turn them into logprobs.
        '''
        # Hint: you can assign a new value to label to feature_probs or do the normalisation in place
        # for each label we have to go through each counter
        for label in self.feature_probs.keys():
            self.label_feature_log_probs[label] = dict()
            # for each counter we have to get the total count and assign the log-prob to the feature
            feature_counter = self.feature_probs[label].most_common()
            total_features = reduce(lambda current, count: current+count[1], feature_counter, 0)
            for (x, y) in feature_counter:
                self.label_feature_log_probs[label][x] = math.log2(float(y)/total_features)

    def predict(self, data):
        ''' 
        Predict the most probable label according to the model on a stream of data.
        
        :param data: A stream of string data from which to extract features
        :return: the most probable label for the data
        '''
        total_probs = dict()
        for line in data:
            for feature in line.lower().split():
                if feature not in self.vocabulary:
                    # we only understand our vocabulary
                    continue
                # we try using only features which have some relevance
                average_probability = 0.0
                for label in self.label_log_probs.keys():
                    average_probability += self.label_feature_log_probs[label][feature]
                average_probability /= len(self.label_log_probs)
                good_feature = False
                for label in self.label_log_probs.keys():
                    if self.label_feature_log_probs[label][feature] - average_probability > self.delta:
                        good_feature = True
                        break
                if not good_feature:
                    continue

                for label in self.label_log_probs.keys():
                    if feature in self.label_feature_log_probs[label]:
                        if label not in total_probs:
                            # initialize total probs lazily: log(1) + log(P(Y))
                            total_probs[label] = 0.0 + self.label_log_probs[label]
                        # "multiply" (with logs we do addition) probabilities together
                        #P(Y|X^n)~P(Y)*P(X1|Y)*...*P(Xn|Y)
                        total_probs[label] += self.label_feature_log_probs[label][feature]

        # select the label which has the highest probability
        most_likely_label_value = -math.inf
        most_likely_label = None
        for label in total_probs.keys():
            if total_probs[label] > most_likely_label_value:
                most_likely_label = label
                most_likely_label_value = total_probs[label]
        return most_likely_label
