from collections import Counter

# stores a mapping from code words to characters
encoding_map = dict()
# stores a mapping from code words to characters
decoding_map = dict()
counter = Counter()
total = 0

def construct_code(path):
    '''Construct a code from the empirical distribution found in a text file.


    :param path: The path to the text file
    '''
    global total

    with open(path) as text_file:
        for line in text_file:
            total += len(line)
            counter.update(line)

    # order list according to increasing letter counts
    ordered_list = sorted(counter.most_common(), key=lambda letter: letter[1])

    # initiate encoding_map with empty codewords
    for letter, count in ordered_list:
        encoding_map[letter] = ''

    # build the Huffman tree
    while len(ordered_list)>1:
        # take the two least likely symbols and prepend all symbols in the corresponding subtree with 0 or 1
        for letter in ordered_list[0][0]:
            encoding_map[letter] = '0' + encoding_map[letter]
        for letter in ordered_list[1][0]:
            encoding_map[letter] = '1' + encoding_map[letter]
        # create a new symbol which is the concatenation of the two least likely symbols
        # and the letter count is the sum of the two previous counts
        ordered_list[0] = (ordered_list[0][0] + ordered_list[1][0], ordered_list[0][1]+ordered_list[1][1])
        del ordered_list[1]
        # resort the list
        ordered_list = sorted(ordered_list, key=lambda letter: letter[1])

    # construct the decoding_map
    for letter, count in counter.most_common():
        decoding_map[encoding_map[letter]] = letter


def average_code_word_length():
    '''Compute the average code word length for the Huffman code

    :return: The average code word length
    '''
    # Hint: use the keyword "global" to get access to the dictionaries
    # for item in encoding_map

    word_list = counter.most_common()
    m = 0.0
    p = 0.0
    l = 0.0
    for letter, count in word_list:
        p= (float(count)/total)
        l = len(encoding_map[letter])
        m += p * l
    return m


def encode(some_string):
    '''Encode a string according to a Huffman code.

    :param some_string: The string to encode
    :return: The encoding of the string
    '''
    encoded_string = ""

    for letter in some_string:
        encoded_string += encoding_map[letter]
    return encoded_string





def decode(some_code):
    '''Decode a Huffman-encoded message.

    :param some_code: The code sequence
    :return: The decoded message
    '''
    l2=1
    secret_message=""
    while len(some_code):
        working_code = some_code[0:l2]
        if working_code in decoding_map:
            secret_message += decoding_map[working_code]
            some_code = some_code[l2:]
            l2 = 1
        else:
            l2 += 1
    return secret_message

construct_code("../../../war_and_peace.txt")
print("Average code word length is {}".format(average_code_word_length()))
print(decode(encode("Hello World!")))
print(decode("101100100001100011111000110101101010010010100101101110101000110"
             "111001001101111110010111100101000101000011010111111011111110100011"
             "01001101101010001100011001100001011001110101011000010110011100001"
             "1111111001010000100001100010000001000100100011101101111010010"))
