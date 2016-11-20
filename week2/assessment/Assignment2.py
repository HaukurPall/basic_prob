# note that the program needs 2-3 seconds before it offers you to choose an action after you gave it the path to a text file

path = input("Welcome. Please copy-paste the path to the .txt file of the book that you want to know something about here.\n")

# creating lower-case version of the input
warandpeace = open(path) # read file from your computer
contents = list()
for line in warandpeace:
    contents.append(line)
warandpeace.close()

warandpeace2 = open('warandpeace2.txt', 'w') # writes new file
for line in contents:
    warandpeace2.write(line.lower()) # in lower case

# make a huge dictionary counting all the words that appear with counter
count = {}
warandpeace3 = open('warandpeace2.txt')
for line in warandpeace3:
    for word in line.split():
        key = word
        if key in count:
            count[key] += 1
        else:
            count[key] = 1

sort = sorted(count, key = count.get) # sorted by counts (ascending) but not yet alphabetically


# creating a list that is ordered by counts and then alphabetically
valueDict = {}
maxValue = 0
valueList = []
wordLeast = []
wordMost = []
for key, value in count.items(): # valueDict will be a dictionary where the occurence numbers are the keys and the words the values
    if value > maxValue:
        maxValue = value
    if value not in valueList:
        valueList.append(value)
    if value in valueDict:
        valueDict[value] = valueDict[value] + [key]
    else:
        valueDict[value] = [key]
valueList  = sorted(valueList)
for i in valueList: # list that is ordered for least common. equally common words are orderes alphabetically
    wordLeast = wordLeast + sorted(valueDict[i])
for i in reversed(valueList): # list that is ordered for most common. equally common words are orderes alphabetically
    wordMost = wordMost + sorted(valueDict[i])

print('Thank you. What do you wish to do with this file? Please answer with the number of the action that you want to perform.')

while True:
    action = input("Would you like to:\n 1. Look for the n most frequent words \n 2. Look for the n least frequent words \n 3. Look for the count of a specific word \n 4. Exit\n")

    if action == '1': # most frequent words
        action2 = input('The how many most frequent words would you like to now? Please answer with an integer. Words that occur equally often will be ordered alphabetically.\n')
        try:
            val = int(action2) # check if input has correct form
            print('The ' + action2 + ' most frequent words are:\n')

            for i in range(0, int(action2)):
                print(str(i+1) + ') ' + wordMost[i] + ': ' + str(count[wordMost[i]]) + ' times')

        except ValueError: # in case input does not have correct form
            print('That is not an integer. Please try another action.')

    elif action == '2': # least frequent words
        action2 = input('The how many least frequent words would you like to now? Please answer with an integer. Please note that there is usually plenty of words that occur only once. Words that occur equally often will be ordered alphabetically. \n')
        try:
            val = int(action2) # check if input has correct form
            print('The ' + action2 + ' least frequent words are:\n')

            for i in range(0, int(action2)):
                print(str(i+1) + ') ' + wordLeast[i] + ': ' + str(count[wordLeast[i]]) + ' times')

        except ValueError: # in case input does not have correct form
            print('That is not an integer. Please try another action.')

    elif action == '3': # occurences of one word
        action2 = input('Alright. Type in the word you wish to know the count of. Please use only lowercase letters.\n')
        if action2 in count.keys():
            k = str(count[action2])
            print('The word "' + action2 + '" appers ' + k + ' times.')
        else:
            print('Sorry, this word does not occur in the text.')
    elif action == '4': # exit
        break

    else: # input incorrect
        print('Sorry, I do not understand this. Please choose again. The only valid inputs are the integers 1, 2, 3, and 4.')

print('It was a pleasure talking with you. Bye-Bye.')





