from functools import reduce


def listSum(array, total, index):
    if index == len(array):
        return total
    total += array[index]
    return listSum(array, total, index + 1)


total = 0
array = [1, 2, 3, 4, 6, 6]
print(listSum(array, total, 0))

print(reduce(lambda x, y: x+y, array, total))
