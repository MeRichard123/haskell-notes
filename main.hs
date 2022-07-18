import Data.List
-- import Data.List hiding (group)
-- import everything apart from group
-- import Data.List as NewName
-- importing data types:
-- data DataType = A | B | C
-- import Module (DataType(..))
import MyModule (doubleList)

-- Check if number is in a range
inRange :: Integer -> Integer  -> Integer -> Bool
inRange min max x = x >= min && x <= max

fac :: Integer -> Integer
fac n
    | n <= 1 = 1
    | otherwise = n * fac (n - 1)

-- sum of a list
listSum :: Num p => [p] -> p
listSum [] = 0                      -- base case
listSum (x:xs) = x + listSum xs     -- recursive case

reverseList :: [a] -> [a]
reverseList [] = []                  -- base case
reverseList (x:xs) = reverseList xs ++ [x] -- recursive case
-- x:xs is a pattern matching x is the fist element and xs is the rest of the list
-- we then call reverseList on the rest of the list and add the first element to the end of the list

-- lambda expression to add two numbers
add :: Integer -> Integer -> Integer
add = (\x -> (\y -> x + y))

-- return the count of a given element in a list
count :: (Foldable t, Eq a1, Num a2) => a1 -> t a1 -> a2
-- reduce using an accumulator function and an initial value of 0
count e = foldr (\x acc -> if e == x then acc + 1 else acc) 0


-- data types
type Color = (Integer, Integer, Integer)
type Palette = [Color]
-- newtype is a data type that is a single constructor
newtype Name = Name String
-- new type is restricted to one field (isomorphic)


-- quick sort
-- using list comprehensions
qsort :: Ord a => [a] -> [a]
qsort []     =  []
qsort (x:xs) = qsort smaller ++ [x] ++ qsort larger
               where
                   smaller = [a | a <- xs, a <= x]
                   larger  = [b | b <- xs, b > x]

-- io is a monad that can be used to perform IO actions
main :: IO ()
main = do
    -- putStrLn prints a string to the console with a new line
    -- putStr prints a string to the console without a new line 
    putStrLn "Hello World"
    if inRange 0 10 15 then
        putStrLn "In Range"
    else
        putStrLn "Out of Range"
    -- Calling all the functions above
    print(fac 8)
    print(listSum [1,2,3,4,6,6])
    -- reverse is the built in function
    print(reverse [1,2,3,4,5,6,6])
    print(reverseList [1,2,3,4,5,6,6])
    -- Increment all values in a list
    print(map (\x -> x + 1) [1,2,3,4,5,6])
    -- Even is built in basically x % 2 == 0 
    -- (\x x `mod` 2 == 0)
    -- can do filter (\x even x) []
    -- refactor to:
    print(filter even [1,2,3,4,5,6,7,8,9])
    -- Calling the above lambda
    print(add 1 2)
    -- Folding
    print(foldr (+) 0 [1,2,3,4,5,6,7,8])
    print(count 1 [1,2,3,4,1,1,1,1])
    -- input and output using getLine and putStrLn in haskell
    putStrLn "What is your name?"
    name <- getLine
    putStrLn ("Hello " ++ name ++ ".")
    print(doubleList [1,2,3,4,5,6,7,8,9])
    
    print(qsort [1,80,-9,5,7,8,5,15301,17])
-- exampleMain :: IO ()
-- exampleMain = do
--     i <- getLine
--     if i /= "quit" then do
--         putStrLn ("Input:" ++ i)
--         exampleMain
--     else
--         return ()

