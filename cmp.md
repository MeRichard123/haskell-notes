```py
from functools import reduce
reduce(lambda x, y: x+y, array, 0)
```



```py
def listSum(array):
    if len(array) == 0:
        return 0
    return array[0] + listSum(array[1:])
```

```py
listSumTwo = lambda array: array[0] + listSumTwo(array[1:]) if array else 0
```

```hs
(foldr (+) 0 [1,2,3,4,5,6,7,8])
```