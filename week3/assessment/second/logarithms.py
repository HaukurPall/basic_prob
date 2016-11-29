import math

def log_add(a,b):
    '''Adds to numbers in their logarithmic transformtions.

    :param a: The first logarithmically transformed number.
    :param b: The second logarithmically transformed number.
    :return: The log-sum of the two numbers
    '''
    if b == -math.inf:
        return a
    if a == -math.inf:
        return b

    return a + math.log1p(math.exp(b-a))



def log_add_list(list_of_numbers):
    '''Adds all the logarithmically transformed numbers in a list.

    :param list_of_numbers: A list of logarithmically transformed numbers.
    '''
    wert = 0.0
    for i in range(0, len(list_of_numbers), 2):
        wert += log_add(list_of_numbers[i-1], list_of_numbers[i])
    return wert

def log_subtract(a , b = -math.inf):
    '''Subtracts a logarithmically transformed number b from another such number a.

    :param a: The first logarithmically transformed number.
    :param b: The second logarithmically transformed number.
    :return: The log-difference between a and b
    '''
    if a == b:
        return print("-inf")
    if math.isinf(a):
        return b
    if math.isinf(b):
        return a
    return a + math.log1p(-math.exp(b-a))


def log_subtract_list(list_of_numbers):
    '''Subtracts all the logarithmically transformed numbers in a list from the first one.

    :param list_of_numbers: A list of logarithmically transformed numbers.
    '''
    warzenschwein = list_of_numbers[0]
    for i in range(1, len(list_of_numbers)):
        warzenschwein -= log_subtract(list_of_numbers[i])
    return warzenschwein

