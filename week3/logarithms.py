import math

from functools import reduce


def log_add(c, d):
    """Adds two numbers in their logarithmic transformtions.

    :param c: The first logarithmically transformed number.
    :param d: The second logarithmically transformed number.
    :return: The log-sum of the two numbers
    """
    if type(c) != float or type(c) != float:
        return "Error! Input to log_add is not float {} + {}".format(c, d)
    if c == -math.inf:
        return d
    if d == -math.inf:
        return c
    # we have c = log(a) and d = log(b). We want to compute log(a + b)
    # To do this we compute c + log(1 + exp(d − c)) as suggested in script.
    if c > d:
        return c + math.log1p(math.exp(d - c))
    else:
        return d + math.log1p(math.exp(c - d))


def log_add_list(list_of_numbers):
    """Adds all the logarithmically transformed numbers in a list.

    :param list_of_numbers: A list of logarithmically transformed numbers.
    """
    return reduce((lambda x, y: log_add(x, y)), list_of_numbers, -math.inf)


def log_subtract(c, d):
    """Subtracts a logarithmically transformed number d from another such number c.

    :param c: The first logarithmically transformed number.
    :param d: The second logarithmically transformed number.
    :return: The log-difference between c and d
    """
    if type(c) != float or type(c) != float:
        return "Error! Input to log_subtract is not float {} + {}".format(c, d)
    if c == -math.inf:
        return d
    if d == -math.inf:
        return c
    # we have c = log(a) and d = log(b). We want to compute log(a + b)
    # To do this we compute c + log(1 - exp(d − c)) as suggested in script.
    if c > d:
        return c + math.log1p(-math.exp(d - c))
    elif c == d:
        return -math.inf
    else:
        return "Error! Input to log_subtract {} < {}".format(c, d)


def log_subtract_list(list_of_numbers):
    """Subtracts all the logarithmically transformed numbers in a list from the first one.

    :param list_of_numbers: A list of logarithmically transformed numbers.
    """
    if len(list_of_numbers) == 0:
        return -math.inf
    if len(list_of_numbers) == 1:
        return list_of_numbers[0]
    first_value = list_of_numbers.pop(0)
    rest_summed_up = log_add_list(list_of_numbers)
    return log_subtract(first_value, rest_summed_up)


# My tests
print("Addition")
print(str( math.log(0.5)))
logs = [math.log(0.5), math.log(0.5), -math.inf]
print(log_add(logs[0], logs[1]))
print(log_add_list(logs))
logs = [math.log(0.5), math.log(0.5), math.log(0.5), math.log(0.5), math.log(0.5)]
print(log_add_list(logs))
logs = [-math.inf]
print(log_add_list(logs))
logs = []
print(log_add_list(logs))
print("Subtraction")
logs = [math.log(0.5), math.log(0.5), -math.inf]
print("be -inf")
print(log_subtract(logs[0], logs[1]))
print(log_subtract(logs[1], logs[2]))
print("be -inf")
print(log_subtract_list(logs))
logs = [math.log(0.5), -math.inf]
print(log_subtract_list(logs))
print("be -inf")
logs = [math.log(2.0), math.log(0.5), math.log(0.5), math.log(0.5), math.log(0.5)]
print(log_subtract_list(logs))
logs = [-math.inf]
print(log_subtract_list(logs))
logs = []
print(log_subtract_list(logs))
