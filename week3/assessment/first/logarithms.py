import math


def log_add(a, b):
    '''Adds to numbers in their logarithmic transformtions.

    :param a: The first logarithmically transformed number.
    :param b: The second logarithmically transformed number.
    :return: The log-sum of the two numbers
    '''

    largest_value = max(a,b) #largest of two arguments.
    smallest = min(a, b) #smallest of two arguments.
    # when is log(0).
    if math.exp(smallest) == 0:
        return largest_value

    else:
        return largest_value + math.log1p(math.exp(smallest - largest_value))

def log_add_list(list_of_numbers):
    '''Adds all the logarithmically transformed numbers in a list.

    :param list_of_numbers: A list of logarithmically transformed numbers.
    '''

    log_sum_list = list_of_numbers[0]
    for x in list_of_numbers [1:]:
        log_sum_list = log_add(log_sum_list, x)
    return log_sum_list



def log_subtract(a , b):
    '''Subtracts a logarithmically transformed number b from another such number a.

    :param a: The first logarithmically transformed number.
    :param b: The second logarithmically transformed number.
    :return: The log-difference between a and b
    '''
    if a > b :
        return a + math.log1p(- math.exp(b - a))
    else:
        return print("Invalid subtraction ")



def log_subtract_list(list_of_numbers):
    '''Subtracts all the logarithmically transformed numbers in a list from the first one.

    :param list_of_numbers: A list of logarithmically transformed numbers.
    '''
    log_sub_sum_list = list_of_numbers[0]
    for x in list_of_numbers[1:]:
        log_sub_sum_list = log_subtract(log_sub_sum_list,x)
    return log_sub_sum_list



