-- 1. Reverse An Array
rev :: [a] -> [a]
rev = foldl (\acc x -> x : acc) []
-- Alternate solution if x and acc where in a different order we can use flip with partial function
-- application.
rev2 :: [a] -> [a]
rev2 = foldl (flip (:)) []

-- 2. Return a list of prefixes
prefixes :: [a] -> [[a]]
prefixes = foldr (\x acc -> [x] : map ((:) x) acc) []

-- 3. given a set of k+1 data points (x0,y0),...,(xn,yn) 
-- where no two xn are the same. Create an Interpolation Polynomial in the Lagrange form
lagrange :: [(Float, Float)] -> Float -> Float 
lagrange xs x = foldl (\acc (xj,y) -> acc + (y * l xj)) 0 
    xs 
        where
            l xj = foldl (
                \acc (xk, _) ->
                    if xj == xk then
                        acc
                    else
                        acc * ((x-xk)/(xj-xk))
                ) 1 xs    

-- 3. Post-Order Traversal
data Trie a = Leaf a | Node a [Trie a]
foldtrie :: (b -> a -> b) -> b -> Trie a -> b
foldtrie f acc (Leaf x) = f acc x
foldtrie f acc (Node x xs) = foldl f' (f acc x) xs
    where
        f' acc t = foldtrie f acc t

main :: IO()
main = do
    putStr "Hey"
    print(rev [0,1,2,3,4,5])
    print(rev2 [0,1,2,3,4,5])
    print(prefixes [1..15])
    print(lagrange [(1.0,1.0),(2.0,2.0),(5.0,8.0)] 0)