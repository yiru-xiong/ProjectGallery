MIN_LINE_LENGTH = 3
MIN_SENTENCE_LENGTH = 3

def preprocess_input(sentence_tokenizer, file_name):
    preprocessed_content = ''

    with open(file_name, 'r', encoding='utf-8') as input_file:
        all_content = input_file.read()

    # assume all sentences would not be affected by newline in the middle
    for line in all_content.split('\n'):
        if not line.endswith('.') \
                and not line.endswith('?') \
                and not line.endswith('!') \
                and not line.endswith('"') \
                and not line.endswith("'") \
                and not line.endswith(';') \
                and not line.endswith(')'):
            continue

        if len(line) <= MIN_LINE_LENGTH:
            continue

        for sentence in sentence_tokenizer(line).sents:
            if len(sentence) < MIN_SENTENCE_LENGTH:
                continue

            if len(preprocessed_content) != 0:
                preprocessed_content += ' ' + str(sentence)
            else:
                preprocessed_content += str(sentence)

    return preprocessed_content
