
recSum :: Num a => [a] -> a
recSum [] = 0
recSum (x:xs) = (2*x) + recSum xs

recProd :: Num a => [a] -> a
recProd [] = 1
recProd (x:xs) = (2*x) * recProd xs



main :: IO()
main = do
    print(sum $ map (* 2) [1..10])
    print(foldr (\x -> (+) (2 * x)) 0 [1..10])
    print(recSum [1..10])
    
    print(product $ map (* 2) [1..10])
    print(foldr (\x -> (*) (2 * x)) 1 [1..10])
    print(recProd [1..10])