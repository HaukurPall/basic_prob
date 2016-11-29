# What follows is the code for the Huffman-part of the assignment.

from collections import Counter

# Stores a mapping from code words to characters
encoding_map = dict()
# Stores a mapping from code words to characters
decoding_map = dict()
counter = Counter()
total = 0

# Define construction function construct_code
def construct_code(path):
    '''Construct a code from the empirical distribution found in a text file.


    :param path: The path to the text file
    '''
    global total
    with open(path) as text_file:
        for line in text_file:
            total += len(line)
            counter.update(line)

    # Order list according to increasing letter counts
    ordered_list = sorted(counter.most_common(), key=lambda letter: letter[1])

    # Initiate encoding_map with empty codewords
    for letter, count in ordered_list:
        encoding_map[letter] = ''

    # Build the Huffman tree
    while len(ordered_list)>1:
        # take the two least likely symbols and prepend all symbols in the corresponding subtree with 0 or 1
        for letter in ordered_list[0][0]:
            encoding_map[letter] = '0' + encoding_map[letter]
        for letter in ordered_list[1][0]:
            encoding_map[letter] = '1' + encoding_map[letter]
        # Create a new symbol which is the concatenation of the two least likely symbols
        # And the letter count is the sum of the two previous counts
        ordered_list[0] = (ordered_list[0][0] + ordered_list[1][0], ordered_list[0][1]+ordered_list[1][1])
        del ordered_list[1]
        # Resort the list
        ordered_list = sorted(ordered_list, key=lambda letter: letter[1])

    # Construct the decoding_map
    for letter, count in counter.most_common():
        decoding_map[encoding_map[letter]] = letter


# Define function to compute average code length of above Huffman encoding (Note: this function will only work if used in this specific script here)
def average_code_word_length():
    '''Compute the average code word length for the Huffman code

    :return: The average code word length
    '''
    # Use counter from construct_code to count symbols
    symbols_count = counter.most_common()
    # Initialize variable
    sum_over_symbols = 0.0
    # Get the total number of symbols from construct_code. It's called "total" and is a global variable.
    # Compute the average code word length
    # Average code word length = sum over all symbols(probability of symbol*length of codeword for that symbol), probability=count/total
    for letter, count in symbols_count:
        sum_over_symbols += (float(count)/total)*len(encoding_map[letter])
    return sum_over_symbols

print(average_code_word_length()) # For debugging purposes

# Define encoding function using the encoding_map from construct_code. (Note: this function will only work if used in this specific script here)
def encode(some_string):
    '''Encode a string according to a Huffman code.

    :param some_string: The string to encode
    :return: The encoding of the string
    '''
    codelist = []       # List to be filled with encodings of symbols
    for x in some_string:
        codelist.append(encoding_map[x])
    code = "".join(codelist)
    return code

# Define decoding function using the decoding_map from construct_code. (Note: this function will only work if used in this specific script here)
def decode(some_code):
    '''Decode a Huffman-encoded message.

    :param some_code: The code sequence
    :return: The decoded message
    '''
    # Concatenate code one symbol at a time to a string (possible_codeword).
    # Check for each new symbol added if there is a corresponding codeword in the dictionary.
    # If there is a such a codeword, add the symbol it maps to to a new string (decoded_message).
    # Start forming a new possible codeword by adding the next symbol in the code.
    # This is possible because we know that Huffman codes are prefix-free.
    possible_codeword = ""
    decoded_message = ""
    for x in range(0, len(some_code)):
        possible_codeword += str(some_code[x])
        if possible_codeword in decoding_map:
            decoded_message += str(decoding_map[possible_codeword])
            possible_codeword = ""
    return decoded_message

construct_code("../../../war_and_peace.txt")
print("Average code word length is {}".format(average_code_word_length()))
print(encode("Hello World!"))
print(decode("101100100001100011111000110101101010010010100101101110101000110"
             "111001001101111110010111100101000101000011010111111011111110100011"
             "01001101101010001100011001100001011001110101011000010110011100001"
             "1111111001010000100001100010000001000100100011101101111010010"))
