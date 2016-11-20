import collections
from operator import itemgetter

c = collections.Counter()

#file_path = input("Complete file-path, please: ")
file_path = "war_and_peace.txt"
with open(file_path) as input_file:
    for line in input_file:
        c.update(line.lower().split())
number_unique_words = len(c.most_common())
command = ""
list_of_words = c.most_common()
# first we order after alphabetical
list_of_words.sort(key=itemgetter(0))
# then we order, in place after count
list_of_words.sort(key=itemgetter(1), reverse=True)

list_of_words = c.most_common()[-1000:]
list_of_words.sort(key=itemgetter(0))
# then we order, in place after count
list_of_words.sort(key=itemgetter(1), reverse=True)
for i in range(0, 1000):
    print(list_of_words[i][0] + "\t" + str(list_of_words[i][1]))

while command != "4":
    print("What would you like to do? Press the associated number for the action:")
    print("1. Look for the most frequent words")
    print("2. Look for the least frequent words")
    print("3. Look for the count of a specific word")
    print("4. Exit")
    command = input()

    if command == "1":
        number_of_words = int(input("How many words: "))
        for i in range(0, number_of_words):
            print(list_of_words[i][0] + "\t" + str(list_of_words[i][1]))
    elif command == "2":
        number_of_words = int(input("How many words: "))
        list_of_words = list_of_words[number_unique_words - number_of_words:]
        for i in range(0, number_of_words):
            print(list_of_words[i][0] + "\t" + str(list_of_words[i][1]))
    elif command == "3":
        word = input("What word: ")
        if word in c:
            print(word + "\t" + str(c[word]))
        else:
            print("Sorry, this word does not occur in the text")
    elif command == "4":
        print("Good bye!")
    else:
        print("Sorry, I do not understand this. Please choose again.")
