# flake8: noqa
from functools import reduce


def listSum(array):
    if len(array) == 0:
        return 0
    return array[0] + listSum(array[1:])

listSumTwo = lambda array: array[0] + listSumTwo(array[1:]) if array else 0 


array = [1, 2, 3, 4, 6, 6]
print(listSum(array))

print(reduce(lambda x, y: x+y, array, 0))
