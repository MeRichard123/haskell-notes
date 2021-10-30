import Data.List

inRange :: Integer -> Integer  -> Integer -> Bool
inRange min max x = x >= min && x <= max

fac :: Integer -> Integer
fac n
    | n <= 1 = 1
    | otherwise = n * fac (n - 1)

listSum :: Num p => [p] -> p
listSum [] = 0
listSum (x:xs) = x + listSum xs

reverseList :: [a] -> [a]
reverseList [] = []
reverseList (x:xs) = reverseList xs ++ [x]

add :: Integer -> Integer -> Integer
add = (\x -> (\y -> x + y))

count :: (Foldable t, Eq a1, Num a2) => a1 -> t a1 -> a2
count e = foldr (\x acc -> if e == x then acc + 1 else acc) 0


main :: IO ()
main = do 
    putStr "Hello World\n"
    if inRange 0 10 15 then
        putStr "In Range\n"
    else
        putStr "Out of Range\n"
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

    