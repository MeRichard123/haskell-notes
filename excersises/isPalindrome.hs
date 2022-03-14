isPalindrome :: String -> Bool
isPalindrome string 
  | reverse string == string = True
  | otherwise = False


isPalindrome2 :: String -> Bool
isPalindrome2 s = s == reverse s


recursiveReverse :: [a] -> [a]
recursiveReverse [] = []
recursiveReverse (x:xs) = recursiveReverse xs ++ [x]

isPalindromeRecursive :: String -> Bool
isPalindromeRecursive s = recursiveReverse s == s


main :: IO()
main = do
    print(isPalindrome "racecar")
    print(isPalindrome2 "racecar")
    print(isPalindromeRecursive "racecar")
