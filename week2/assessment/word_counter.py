"""
Counts the occurrances of words program

1. Paste the path to your text file (with the filename and extension)
2...Select program action with the number keys, as prompted by the program

Basic Probability: week 2
16-11-2016
"""
import collections


filename = input("path and name of the textfile: ")

# The counter object which will keep track of occurrences of each word
wordCounter = collections.Counter()

# Counting all word occurrences in the book
with open(filename, 'r') as file:
    for line in file:
        line = line.lower().split()
        wordCounter.update(line)


accepted = [1,2,3,4] # Accepted user inputs
# Request input from user and keep executing requests
while True:
    # Input prompt print
    userInput = int(input("\nWhat would you like to do?\n"
                      "1. Look for the most frequeny words\n"
                      "2. Look for the least frequent words\n"
                      "3. Look for the count of a specific word\n"
                      "4. Exit\n"))

    # Retry input if input is not in accepted list
    while userInput not in accepted:
        userInput = int(input("Sorry, I do not understand this. Please choose again: "))

    # Show most frequent words (option 1)
    if userInput == 1:
        n = int(input("Please state how long the wordlist should be:"))
        commonElements = wordCounter.most_common(n)
        # First sort on negative value of second tuple item of commonElements (descending sort results)...
        # secondly sort on first item (alphabet)
        commonElements.sort(key=lambda Item: (-Item[1], Item[0]))
        for i in range(n):
            print("{0:<9} : {1}".format(*commonElements[i])) # * splits the tuple into different args

    # Show least frequent words (option 2)
    elif userInput == 2:
        n = int(input("Please state how long the wordlist should be:"))
        rareElements = wordCounter.most_common()
        rareElements = rareElements[-n:] # Taking the the end of the most common list = least common
        rareElements.sort(key=lambda Item: (-Item[1], Item[0]))
        for i in range(n):
            print("{0:<9} : {1}".format(*rareElements[i])) # {:<9} says that first argument takes min. space of 9 chars

    # Show count of requested word (option 3)
    elif userInput == 3:
        requested = input("Please type the word you want to know the number of occurrances of: ").lower()
        occurences = wordCounter[requested]
        if occurences >= 1:
            print("{} : {}\n".format(requested,occurences))
        else:
            print("'{}' does not occur in the text".format(requested))

    else: # EXIT (option 4)
        print("\nGoodbye!\n")
        break