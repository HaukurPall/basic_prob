from collections import Counter
from functools import reduce
from math import log2
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
        self.label_log = dict()
        self.label_feature_log = dict()

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

        # To make a set of all words that occur
        for word in features:
            self.vocabulary.add(word)

            # To make a counter of the Label within the dictionary
            if label not in self.feature_probs:
                self.feature_probs[label] = Counter()

            # Put the words in the corresponding counter of the label
            self.feature_probs[label].update([word])

    def smooth_feature_counts(self, smoothing=1):
        '''Smooth the collected feature counts

        :param smoothing: The smoothing constant
        '''

        # This is the set of all our words that occur.
        for word in self.vocabulary:

            for label in self.feature_probs:

                # To give positive probability to all the words that occur in our set.
                if word in self.feature_probs[label]:
                    self.feature_probs[label][word] += smoothing

                # Else not.
                else:
                    self.feature_probs[label][word] = smoothing

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

        # To get a total amount n for the labels:
        n_labels = reduce(lambda current, count: current + count[1], (self.label_probs.most_common()), 0)

        for (a, b) in self.label_probs.most_common():

            # To calculate the probability:
            self.label_log[a] = log2(float(b) / (n_labels))

    def log_normalise_feature_probs(self):
        '''
        Normalize the feature counts for each label to probabilities and turn them into logprobs.
        '''
        # Hint: you can assign a new value to label to feature_probs or do the normalisation in place


        for label in self.feature_probs.keys():

            self.label_feature_log[label] = dict()

            # for each counter we have to get the total count and assign the log-prob to the feature
            feature_counter = self.feature_probs[label].most_common()

            # To get a total amount n for the features:
            n_features = reduce(lambda current, count: current + count[1], feature_counter, 0)

            for (a, b) in feature_counter:

                # To calculate the probability:
                self.label_feature_log[label][a] = log2(float(b) / (n_features))


    def predict(self, data):
        ''' 
        Predict the most probable label according to the model on a stream of data.
        
        :param data: A stream of string data from which to extract features
        :return: the most probable label for the data
        '''
        total_probability = dict()

        for line in data:

            for word in line.lower().split():

                if word not in self.vocabulary:

                    continue

                for label in self.label_log.keys():

                    if word in self.label_feature_log[label]:

                        if label not in total_probability:

                            total_probability[label] = 0.0 + self.label_log[label]

                        # Application of the chain rule and adding the probabilities:

                        total_probability[label] += self.label_feature_log[label][word]

        lll = -999999999.9

        likeliest_label = None

        for label in total_probability.keys():

            # Give the data the label which has the highest probability:

            if total_probability[label] > lll:

                likeliest_label = label

                lll = total_probability[label]

        # Give the likeliest label according to the data in the output.

        return likeliest_label
