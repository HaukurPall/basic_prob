from collections import Counter
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

    def train(self, data, label):
        '''
        Train the classifier by counting features in the data set.
        
        :param data: A stream of string data from which to extract features
        :param label: The label of the data
        '''
        for x in data:
            self.add_feature_counts(x.split(), label)
    
    def add_feature_counts(self, features, label):
        '''
        Count the features in a feature list.
        
        :param features: a list of features.
        :param label: the label of the data file from which the features were extracted.
        '''
        # TODO: implement this!
        self.update_label_count(label)

        #add words to vocabulary
        for words in features:
            self.vocabulary.add(words)

            try:
                self.feature_probs[label].update({words: 1})
            except:
                self.feature_probs.update({label: Counter({words: 1})})


    def smooth_feature_counts(self, smoothing=1):
        '''Smooth the collected feature counts

        :param smoothing: The smoothing constant
        '''
        #set the param smoothing to words that did not appear at the label. 
        for label in self.feature_probs:
            for words in self.vocabulary:
                self.feature_probs[label].update({words: smoothing})
        

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
        #
        for label in self.label_probs:
            self.label_probs[label] = math.log((self.label_probs[label]/ sum(self.label_probs.values())))
            
    def log_normalise_feature_probs(self):
        '''
        Normalize the feature counts for each label to probabilities and turn them into logprobs.
        '''
        # Hint: you can assign a new value to label to feature_probs or do the normalisation in place
        
        for label in self.feature_probs:
            features_sum = sum(self.feature_probs[label].values())

            #prob of feature given label.
            for words in self.vocabulary:
                self.feature_probs[label][words] = math.log(self.feature_probs[label][words]/features_sum)

                
    def predict(self, data):
        ''' 
        Predict the most probable label according to the model on a stream of data.
        
        :param data: A stream of string data from which to extract features
        :return: the most probable label for the data
        '''
        # TODO: implement this!
        # list of features
        features = []
        # list of probable label
        probable = {}

        #looping through the data
        for x in data:
            features += x.split()
        
        
        for label in self.label_probs.keys():
            prob = self.label_probs[label]

            for words in features:
                prob = prob + self.feature_probs[label][words]
                
            probable.update({label: prob})

        #sorting     
        probable_sorted = sorted(probable.items(), key = lambda x: x[1], reverse = True)
        return probable_sorted[0][0]







