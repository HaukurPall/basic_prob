from collections import Counter
from math import log, exp, log1p, fsum
import string

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
        self.posterior = dict()
        # create variable to count words in vocabulary
        self.vocabularycount = Counter()
        # create list in self which will later consist of the top 500 most common words
        self.top500 = []

    def train(self, data, label):
        '''
        Train the classifier by counting features in the data set.
        
        :param data: A stream of string data from which to extract features
        :param label: The label of the data
        '''

        # lower the words and get rid of punctuation
        for line in data:
            # get rid of puntuation in words and lower all letters in the text
            line = "".join(char for char in line if char not in string.punctuation)
            line = line.split()
            line = [x.lower() for x in line]
            self.add_feature_counts(line, label)

    def add_feature_counts(self, features, label):
        '''
        Count the features in a feature list.
        
        :param features: a list of features (for each sentence the words in a list).
        :param label: the label of the data file from which the features were extracted.
        (a label assigning a sentence to a text)
        '''
        # if for the incoming label no counter in feature_probs exists, create one
        if self.feature_probs.get(label) is None:
            self.feature_probs[label] = Counter()
        # update the incoming features in the labeled counter in feature_probs
        self.feature_probs[label].update(features)

        # update the vocabulary set
        self.vocabulary.update(features)
        # update the counts of the words in the vocabulary
        self.vocabularycount.update(features)

    def smooth_feature_counts(self, smoothing=1):
        '''Smooth the collected feature counts

        :param smoothing: The smoothing constant
        '''
        for word in self.vocabulary:
            word_list = [word] * smoothing
            # update the words in all labels!
            for label in self.feature_probs.keys():
                self.feature_probs[label].update(word_list)

        
    def update_label_count(self,label):
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

        total_labels = sum(self.label_probs.values())
        for key, value in self.label_probs.items():
            self.label_probs[key] = log((value/total_labels))


    def log_normalise_feature_probs(self):
        '''
        Normalize the feature counts for each label to probabilities and turn them into logprobs.
        Create list with 500 most common words in the vocabulary.
        '''

        # update the feature_probs for all labels!
        for label in self.feature_probs.keys():
            total_counts = sum(self.feature_probs[label].values())
            for key, value in self.feature_probs[label].items():
                self.feature_probs[label][key] = log((value/total_counts))

        # create list wist 500 most common words in vocabulary
        self.top500 = self.vocabularycount.most_common(500)
        # get rid of the counts, we only need a list with the words!
        self.top500 = [x[0] for x in self.top500]

    def predict(self, data):
        ''' 
        Predict the most probable label according to the model on a stream of data.
        
        :param data: A stream of string data from which to extract features
        :return: the most probable label for the data
        '''
        # loop over the 20 labels
        for label in self.label_probs.keys():
            posteriorlist = []
            # for each label create prior
            prior = self.label_probs[label]
            # loop over the data
            for line in data:
                # get rid of punctuation in test text and lower words
                line = "".join(char for char in line if char not in string.punctuation)
                line = line.split()
                line = [x.lower() for x in line]
                for word in line:
                    if word not in self.vocabulary or word in self.top500:
                        # we can not calculate the posterior and therefore continue to the next word
                        continue
                    # word in vocabulary and not most common:
                    else:
                        # save the posterior probability of the words
                        posteriorlist.append(sum([self.feature_probs[label][word], prior]))
            #if the list is empty (no words in vocabulary)
            if len(posteriorlist) == 0:
                continue
            else:
                # sum the probabilities of the words to create the posterior for the label
                self.posterior[label] = sum(posteriorlist)
            # go back to the beginning of the data, to loop multiple times of data
            data.seek(0)
        # the label with the highest probability is most likely to represent the test text
        return max(self.posterior, key = self.posterior.get)
