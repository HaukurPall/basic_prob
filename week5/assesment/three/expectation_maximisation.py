import random, sys
from math import log, exp, factorial
from week5.em.logarithms import log_add, log_add_list

### version 2, modified by Christian Schaffner on Saturday, 3 December 2016


def main():
    '''
    Run the EM-algorithm on the data specified by 'data_path'.
    Initial parameter estimation can either be set to real values, or set to 'None' to generate random parameter values.
    Make sure to provide the same number of parameter values as the number of mixture components.
    '''

    data_path = '../../geometric_example_data.txt'

# Specify number of components
    number_mixture_components = 3

# Specify weight of each component or set to 'None' to generate random weights.
    initial_mixture_weights = [0.3, 0.3, 0.4]
    # initial_mixture_weights = [0.4, 0.4, 0.2]
    # initial_mixture_weights = [0.2, 0.8]

# Specify parameter value of each component or set to 'None' to generate random weights.
    initial_geometric_parameters = [0.2, 0.5, 0.7]
    # initial_geometric_parameters = [0.01, 0.9, 0.2]
    # initial_geometric_parameters = [0.2, 0.6]

    learner = EM(int(number_mixture_components), initial_mixture_weights, initial_geometric_parameters)
    learner.train(data_path, 20)


class GeometricDistribution(object):
    '''An implementation of the geometric distribution that whose support includes 0.'''

    def __init__(self, param=0.5):
        '''Constructor

        :param param: The parameter of this geometric distribution
        :raises: ValueError if param is not in [0,1]
        '''

        self.failure = log(1 - param)
        self.success = log(param)

    def log_prob(self, x):
        '''Compute the log-probability of an observation

        :param x: The observation
        :return: The log-probability of x under this distribution
        :raises: ValueError if x is not a non-negative integer
        '''

        if x%1 != 0 or x < 0:
            raise ValueError("x is not a non-negative integer")

        return self.success + x*self.failure

    def get_param(self):
        '''Get the parameter of this distribution

        :return: The parameter of this distribution
        '''
        return exp(self.success)


class EM(object):
    '''A geometric mixture model that can be trained using EM'''

    def __init__(self, num_components = 2, initial_mixture_weights = None, initial_geometric_parameters = None):
        '''Constructor

        :param num_components: The number of mixture components in this model
        :param initial_mixture_weights: A list of initial mixture weights (weights are initialised randomly if no list is provided)
        :param initial_geometric_parameters: A list of initial component parameters (initialised randomly if no list is provided)
        '''

        # total number of mixture components
        self.num_components = num_components
        # map from components to their mixture_weights (stored as logarithms)
        self.mixture_weights = list()
        # map from components to their distributions
        self.component_distributions = list()
        # map from components to their expected number of occurrence (stored as logarithms)
        self.expected_component_counts = dict()
        # map from components to expected values of observations (stored as logarithms)
        self.expected_observation_counts = dict()
        self.log_likelihood = 0
        self.initialise(initial_mixture_weights, initial_geometric_parameters)

    def initialise(self, initial_mixture_weights = None, initial_geometric_parameters = None):
        '''Initialise the parameters of this model

        :param initial_mixture_weights: A list of initial mixture weights (weights are initialised randomly if no list is provided)
        :param initial_geometric_parameters: A list of initial component parameters (initialised randomly if no list is provided)
        '''

        if initial_mixture_weights == None and initial_geometric_parameters == None:
            for component in range(self.num_components):
                self.mixture_weights.append(random.random())
                self.component_distributions.append(GeometricDistribution(random.random()))
        else:
            for component in range(self.num_components):
                self.mixture_weights.append(initial_mixture_weights[component])
                self.component_distributions.append(GeometricDistribution(initial_geometric_parameters[component]))

            # values that are stored as logarithms are initialized as -inf = log(0)
        for component in range(self.num_components):
            self.expected_component_counts[component] = -float("inf")
            self.expected_observation_counts[component] = -float("inf")

        # normalise priors
        prior_sum = sum(self.mixture_weights)
        for component in range(self.num_components):
            # normalise and transform to log-prob
            self.mixture_weights[component] = log(self.mixture_weights[component] / prior_sum)

    def train(self, data_path, iterations):
        '''Train the model on data for a fixed number of iterations. After each iteration of training, the log-likelihood,
        mixture weights and component parameters are printed out.

        :param data_path: The path to the data file
        :param iterations: The number of iterations
        '''

        for iter in range(iterations):
            params = list()
            priors = list()
            for component in range(self.num_components):
                params.append(self.component_distributions[component].get_param())
                priors.append(exp(self.mixture_weights[component]))

            print("\nIteration {}".format(iter))
            print("log-likelihood: {}".format(self.log_likelihood))
            print("Component priors: {}".format(priors))
            print("Component parameters: {}".format(params))

            # reset log-likelihood
            self.log_likelihood = 0

            self.em(data_path)



    def em(self, data_path):
        '''Perform one iteration of EM on the data

        :param data_path: The path to the data file
        '''
        with open(data_path) as data:
            for line in data:
                for observation in line.split():
                    self.e_step(int(observation))

        self.m_step()


    def e_step(self, observation):
        '''Perform the E-step on a single obervation. That is, compute the posterior of mixture components
        and add the expected occurrence of each component to the running totals
        self.log_likelihood ,
        self.expected_component_counts , and
        self.expected_observation_counts

        :param observation: The observation (x_i)
        '''

        likelihood_list = list()

        for component in range(self.num_components):
            # Prior (= weights)
            log_prior = self.mixture_weights[component]

            # Calculate log-likelihood of one datapoint for each component
            log_likelihood = log_prior + self.component_distributions[component].log_prob(observation)

            # store result in list
            likelihood_list.append(log_likelihood)

        # log sum of likelihood over all components (for this datapoint)
        marginal_likelihood = log_add_list(likelihood_list)

        # Calculate total likelihood of data for this iteration ( = sum of marginal log likelihoods over all datapoints)
        self.log_likelihood += marginal_likelihood

        for component in range(self.num_components):
        # Calculate posterior for current datapoint
            posterior =  likelihood_list[component] - marginal_likelihood

        # Calculate weight of each component ( = cumulative log-addition of posteriors for all datapoints)
            self.expected_component_counts[component] = log_add(self.expected_component_counts[component], posterior)

        # Update expected observation counts
            # Skip if value == 0 (i.e., zero-values won't add anything and will crash log-function)
            if observation == 0:
                continue

            # calculate expected observation count of current datapoint for each component
            value_count_log = log(observation) + posterior

            # add to total observation count of each component
            self.expected_observation_counts[component] = log_add(self.expected_observation_counts[component], value_count_log)


    def m_step(self):
        '''Perform the M-step. This step updates
        self.mixture_weights
        self.component_distributions
        '''

        # test if the sum of the summed expected_component_counts is roughly equal to the total amount of observations
        sum_obs_counts = log_add_list(self.expected_component_counts.values())
        print("Sum of expected component counts: {}".format(exp(sum_obs_counts)))

        param_list = list()
        weight_list = list()
        for component in range(self.num_components):
            # calculate new weight (= expected component counts / total number of datapoints)
            new_weight = exp( self.expected_component_counts[component] - sum_obs_counts )
            weight_list.append(new_weight)

            # for each component, calculate denominator of geometric MLE-function (= expected component counts * expected observation counts )
            # see (https://www.projectrhea.org/rhea/index.php/MLE_Examples:_Exponential_and_Geometric_Distributions_Old_Kiwi)
            nSigmaX = log_add(self.expected_component_counts[component], self.expected_observation_counts[component])

            # calculate new parameter value
            new_param = exp(self.expected_component_counts[component] - nSigmaX)
            param_list.append(new_param)

        for component in range(self.num_components):
            # update weights
            self.mixture_weights[component] = weight_list[component]

            # update parameters
            self.component_distributions[component] = GeometricDistribution(param_list[component])

            # reset counters
            self.expected_component_counts[component] = -float("inf")
            self.expected_observation_counts[component] = -float("inf")



if __name__ == "__main__":
    main()
