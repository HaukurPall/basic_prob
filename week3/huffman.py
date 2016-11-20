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
    while len(ordered_list) > 1:
        # take the two least likely symbols and prepend all symbols in the corresponding subtree with 0 or 1
        for letter in ordered_list[0][0]:
            encoding_map[letter] = '0' + encoding_map[letter]
        for letter in ordered_list[1][0]:
            encoding_map[letter] = '1' + encoding_map[letter]
        # create a new symbol which is the concatenation of the two least likely symbols
        # and the letter count is the sum of the two previous counts
        ordered_list[0] = (ordered_list[0][0] + ordered_list[1][0], ordered_list[0][1] + ordered_list[1][1])
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
    complete_list = counter.most_common()
    sum = 0.0
    for letter, count in complete_list:
        # sum of ( probability * length ), probability = count/total
        sum += (float(count)/total) * len(encoding_map[letter])
    return sum


def encode(some_string):
    '''Encode a string according to a Huffman code.

    :param some_string: The string to encode
    :return: The encoding of the string
    '''
    encoded_string = ""
    if not(type(some_string) is str):
        return "Error! Cannot encode {}, not a string!".format(some_string)
    for letter in some_string:
        if letter not in encoding_map:
            # if the letter does not exist in our encoding map we have to throw an error
            return "Error! Encoding letter {} not found!".format(letter)
        # we add the encoding for the letter to the string we are constructing
        encoded_string += encoding_map[letter]
    return encoded_string


def decode(some_code):
    '''Decode a Huffman-encoded message.

    :param some_code: The code sequence
    :return: The decoded message
    '''
    decoded_string = ""
    current_length = 1
    # we iterate until we have finished cutting all known codes from the string
    while len(some_code) > 0:
        # if we are trying to construct a code which is illegal, we throw and error
        if current_length > len(some_code):
            return "Error! Unable to decode code, no suitable decoding found for code: {}".format(some_code)
        # we always add one more code to our codeword to test if it exists in the map
        current_code = some_code[0:current_length]
        if current_code in decoding_map:
            decoded_string += decoding_map[current_code]
            # we cut off our original code
            some_code = some_code[current_length:]
            # we reset our length
            current_length = 1
        else:
            # we try a larger codeword
            current_length += 1
    return decoded_string


construct_code("../war_and_peace.txt")
print("Average code word length is {}".format(average_code_word_length()))
print(encode("Hello World!"))
print(decode("101100100001100011111000110101101010010010100101101110101000110"
             "111001001101111110010111100101000101000011010111111011111110100011"
             "01001101101010001100011001100001011001110101011000010110011100001"
             "1111111001010000100001100010000001000100100011101101111010010"))
print(decode(encode("Hello World!")))
print(decode(encode("ð")))
