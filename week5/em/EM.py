import random, sys
from math import log, exp, factorial
from week5.em.logarithms import log_add, log_add_list


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

        if x % 1 != 0 or x < 0:
            raise ValueError("x is not a non-negative integer")

        return self.success + x * self.failure

    def get_param(self):
        '''Get the parameter of this distribution

        :return: The parameter of this distribution
        '''
        return exp(self.success)


class EM(object):
    '''A geometric mixture model that can be trained using EM'''

    def __init__(self, num_components=5, initial_mixture_weights=None, initial_geometric_parameters=None):
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

    def initialise(self, initial_mixture_weights, initial_geometric_parameters):
        '''Initialise the parameters of this model

        :param initial_mixture_weights: A list of initial mixture weights (weights are initialised randomly if no list is provided)
        :param initial_geometric_parameters: A list of initial component parameters (initialised randomly if no list is provided)
        '''

        for component in range(self.num_components):
            # initialize with intial parameters if provided
            if initial_mixture_weights:
                self.mixture_weights.append(initial_mixture_weights[component])
            else:
                self.mixture_weights.append(random.random())
            # initialize with intial parameters if provided
            if initial_geometric_parameters:
                self.component_distributions.append(GeometricDistribution(initial_geometric_parameters[component]))
            else:
                self.component_distributions.append(GeometricDistribution(random.random()))
            # values that are stored as logarithms are initialized as -inf = log(0)
            self.expected_component_counts[component] = -float("inf")
            self.expected_observation_counts[component] = -float("inf")

        # normalise priors
        prior_sum = sum(self.mixture_weights)
        for component in range(self.num_components):
            # normalise and tranform to log-prob
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

        :param observation: The observation
        '''
        component_probabilities = list()
        for component in range(self.num_components):
            # prob = weigth * geo(x)
            component_probability = self.mixture_weights[component] + self.component_distributions[component].log_prob(
                observation)
            component_probabilities.append(component_probability)
        # marginal_probabilities:
        observation_probability = log_add_list(component_probabilities)
        self.log_likelihood += observation_probability

        for component in range(self.num_components):
            # component/total
            norm_comp_prob = component_probabilities[component] - observation_probability
            self.expected_component_counts[component] = log_add(self.expected_component_counts[component],
                                                                norm_comp_prob)
            # we take care if observation is 0, we have to define: prob * value = -inf
            if observation == 0:
                self.expected_observation_counts[component] = log_add(self.expected_observation_counts[component],
                                                                      -float("inf"))
            else:
                self.expected_observation_counts[component] = log_add(self.expected_observation_counts[component],
                                                                      norm_comp_prob + log(observation))

    def m_step(self):
        '''Perform the M-step. This step updates
        self.mixture_weights
        self.component_distributions
        '''

        # test if the sum of the summed expected_component_counts is roughly equal to the total amount of observations
        sum_obs_counts = log_add_list(self.expected_component_counts.values())
        print("Sum of expected component counts: {}".format(exp(sum_obs_counts)))
        # sum_obs_values = log_add_list(self.expected_observation_counts.values())
        # print("Sum of expected observational counts: {}".format(exp(sum_obs_values)))

        for component in range(self.num_components):
            self.mixture_weights[component] = self.expected_component_counts[component] - sum_obs_counts
            self.component_distributions[component] = GeometricDistribution(
                exp(
                    self.expected_component_counts[component] -
                    (log_add(self.expected_component_counts[component], self.expected_observation_counts[component]))
                )
            )

            self.expected_component_counts[component] = -float("inf")
            self.expected_observation_counts[component] = -float("inf")


def main(data_path, number_mixture_components, initial_mixture_weights=None, initial_geometric_parameters=None):
    learner = EM(int(number_mixture_components), initial_mixture_weights, initial_geometric_parameters)
    learner.train(data_path, 20)


if __name__ == "__main__":
    #main('../geometric_example_data.txt', 2, [0.2, 0.8], [0.2, 0.6])
    main('../prump.txt', 2, [0.2, 0.8], [0.2, 0.6])
    #main('../geometric_example_data.txt', 3)
