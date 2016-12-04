from math import exp, inf, expm1, log1p

def log_add(a, b):
    '''Adds to numbers in their logarithmic transformations.

    :param a: The first logarithmically transformed number.
    :param b: The second logarithmically transformed number.
    :return: The log-sum of the two numbers
    '''
    if b == -inf:
        return a
    if a == -inf:
        return b
    return a+(log1p((exp(b-a))))

def log_add_list(list_of_numbers):
    '''Adds all the logarithmically transformed numbers in a list.

    :param list_of_numbers: A list of logarithmically transformed numbers.
    '''
    add_log = -inf
    for i in list_of_numbers:
        add_log = log_add(add_log,i)
    return add_log

# n = len(list_of_numbers)
#
# int(list_of_numbers[0]) + (log1p(exp([0]-[])))




def log_subtract(a , b):
    '''Subtracts a logarithmically transformed number b from another such number a.

    :param a: The first logarithmically transformed number.
    :param b: The second logarithmically transformed number.
    :return: The log-difference between a and b
    '''
    return a-(log1p(-(exp(b-a))))

def log_subtract_list(list_of_numbers):
    '''Subtracts all the logarithmically transformed numbers in a list from the first one.

    :param list_of_numbers: A list of logarithmically transformed numbers.
    '''
    sub_log = 0
    for i in list_of_numbers:
        add_log = log_subtract(sub_log,log1p(i))
    return sub_log
