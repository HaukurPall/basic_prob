def my_math_operator(number):
    intermediate = do_something(3)
    return do_something_else(intermediate) + square(3)


def square(number):
    return number**3


def do_something(number):
    return (5 + number)/4


def do_something_else(number):
    return do_something(number) * 4


print(str(my_math_operator(3)))
print(len(str(int(str(len(str(len("Hello"*5)))**10)))))
