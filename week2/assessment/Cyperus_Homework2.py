# Tot use counter:

import collections

from collections import Counter

#starting point:

c = collections.Counter()

#To use itemgetter:

from operator import itemgetter

# Introduction

intro = 'Welcome to Cyperus.'

print(intro)

# Ask the user for a path + name of text file

path_text_file = input("Please enter the path of the text file (including the file name):\n")

# Read that test file into memory (reading it once and store all relevant information)

with open(path_text_file) as input_file:
    for line in input_file:
        c.update(line.lower().split())

total_number_words = len(list(c)) # list(c) gives list of unique elements from c

# Ask the user what he want to do

task = None

while task != "4":
    task = input("What would you like to do?:\n 1 Look for the most frequent words. \n 2 Look for the least frequent words. \n 3 Look for the count of a specific word. \n 4 Exit. \nPlease give the number of the task: \n")

# 1: Look for the most frequent words
    if task == "1":


# Ask for the number n of most frequent words to produce.
        task_1 = input("How many words would you like to list?\n")
        n_m = int(task_1)

        reply1 = c.most_common(n_m)
# When more than 1 word: order alphabetically

        reply1.sort(key=itemgetter(0))

        reply1.sort(key=itemgetter(1), reverse=True)

        for i in range(0, n_m):
            print(reply1[i][0] + "\t" + str(reply1[i][1]))


# 2: Look for the least frequent words
    elif task == "2":

# Ask for the number n of least frequent words to produce.
        task_2 = input("How many words would you like to list?\n")
        n_l = int(task_2)

        reply2 = c.most_common()[:-n_l-1:-1]

        reply2.sort(key=itemgetter(0))

        reply2.sort(key=itemgetter(1), reverse=True)

        for i in range(0, n_l):
            print(reply2[i][0] + "\t" + str(reply2[i][1]))

# When more than 1: order alphabetically

# 3: Look for the count of a specific word
    elif task == "3":
        task_3 = input("From what word would you like the wordcount?\n")
        word = task_3.lower()
        if word in c:

            reply3a = c[word]

            reply3b = task_3+" "+str(reply3a)

            print(reply3b)


# If word does not occur in the text, the program prints:
        else:
            print('Sorry, this word does not occur in the text')



# 4: Exit
    elif task == "4":

        goodbye = 'Goodbye and thank you for making use of Cyperus'

        print(goodbye)

#For invalid information: 'Sorry, I do not understand this. Please choose again', and let the user re-enter the value:
    else:
        print('Sorry, I do not understand this. Please choose again:')

















