
class NormalDistribution(object):
    '''An implementation of the geometric distribution that whose support includes 0.'''

    def __init__(self, median=0.0, variance=0.0):
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

    def train(self):
        # feature * coeff = prediction (continuous features)
        # feature (as categorical) |-> feature_value = ignore default (predictor_0 = 0)
        # feature (as categorical) |-> feature_value = predictor_1 * ceoff_f1_1
        # feature (as categorical) = 1

    def evaluate_model(self):
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
