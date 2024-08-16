import math

no_suffix_space_words = ['$', '-', '€', '£', '"', '(', '–', '[', '{']
no_prefix_space_word = ['-', ',', '?', '.', '!', "'s", ':', '%', '–']

def get_noun_lemma_from_sentence(sentence):
    result = []
    for word in sentence:
        if (word.pos_ == 'NOUN' or word.pos_ == 'PROPN') and word.ent_type_ != 'PERSON':
            result.append(word.lemma_)
    return result


def find_names_in_question(question):
    result = []
    startIndex = -1
    for i, word in enumerate(question):
        if word.ent_type_ == 'PERSON':
            if startIndex != -1:
                continue
            else:
                startIndex = i
        else:
            if startIndex != -1:
                result.append([question[j].text for j in range(startIndex, i)])
                startIndex = -1

    return result

def get_question_list(question_file_path):
    with open(question_file_path, 'r', encoding='utf-8') as question_file:
        question_text = question_file.read()

    question_list = []
    for question in question_text.split('\n'):
        question_list.append(question)
    return question_list

def get_first_valid_index(sentence):
    for i, token in enumerate(sentence):
        if '\n' in str(token):
            return i + 1

    return -1


def capitalize_first_character(word):
    result = ""
    for i, c in enumerate(word):
        if i == 0:
            result = result + c.upper()
        else:
            result = result + c
    return result


def decapitalizeFirstCharacter(word):
    result = ""
    for i, c in enumerate(word):
        if i == 0:
            result = result + c.lower()
        else:
            result = result + c
    return result


def find_all_ner(sentence):
    ner_set = set()
    for ent in sentence.ents:
        ner_set.add(str(ent))
    return ner_set


def find_all_ner_need_capitalized(sentence):
    ner_set = set()
    for ent in sentence.ents:
        if ent.label_ == 'PERSON' or ent.label_ == 'NORP' or ent.label_ == 'FSC' or ent.label_ == 'ORG' \
                or ent.label_ == 'GPE' or ent.label_ == 'LOC' or ent.label_ == 'EVENT' or ent.label_ == 'WORK_OF_ART' \
                or ent.label_ == 'LAW' or ent.label_ == 'LANGUAGE' or ent.label_ == 'DATE':
            for word in ent.text.split(' '):
                ner_set.add(word)

    for token in sentence:
        if token.dep_ == 'compound':
            ner_set.add(token.text)

    return ner_set

# return True if it contains am, is, are, was, were
def contains_aux_for_what(sentence):
    valid_aux = ['am', 'is', 'are', 'was', 'were']
    for token in sentence:
        if token.pos_ == 'AUX' and token.text in valid_aux:
            return True
    return False

def is_be_verb(string):
    be_verb = ['am', 'is', 'are', 'was', 'were']
    if string in be_verb:
        return True
    else:
        return False

# return non-person subjects in ent string format
def find_all_non_person_subj(sentence):
    non_person_subj_tokens = []
    for token in sentence:
        if (token.dep_ == 'nsubj' or token.dep_ == 'compound'or token.dep_ == 'conj') and token.ent_type_ != 'PERSON' and token.pos_ != 'PRON': # avoid asking what question when subject is 'it' (PRON)
            non_person_subj_tokens.append(token.text)

    non_person_subject_ents = []
    for ent in sentence.ents:
        valid = True
        for word in ent.text.split(' '):
            if word not in non_person_subj_tokens:
                valid = False
                break
        if valid:
            non_person_subject_ents.append(ent.text)
    if len(non_person_subject_ents) == 0:
        return non_person_subj_tokens
    else:
        return non_person_subject_ents

# return person subjects in ent string format
def find_all_person_subj(sentence):
    person_subj_token_set = set()

    for token in sentence:
        if (token.dep_ == 'nsubj' or token.dep_ == 'compound' or token.dep_ == 'conj') and token.ent_type_ == 'PERSON':
            person_subj_token_set.add(token.text)

    person_subject_ent_set = set()
    for ent in sentence.ents:
        valid = True
        for word in ent.text.split(' '):
            if word not in person_subj_token_set:
                valid = False
                break
        if valid:
            person_subject_ent_set.add(ent.text)

    return person_subject_ent_set

# return all when-related entities in the sentence
def find_all_time_ent(sentence):
    time_ent_str_set = set()
    for ent in sentence.ents:
        if ent.label_ == 'DATE' or ent.label_ == 'TIME':
            time_ent_str_set.add(ent.text)

    return time_ent_str_set

def find_prep_before_time_ent(sentence, time_ent_str_set):
    time_prep_str = ''
    for token in sentence:
        for time_ent_str in time_ent_str_set:
            if token.text in time_ent_str:
                return time_prep_str
            elif token.dep_ == 'prep':
                time_prep_str = token.text

def find_all_location(sentence):
    location_set = set()
    for ent in sentence.ents:
        if ent.label_ == 'GPE' or ent.label_ == 'LOC':
            location_str = str(ent.text)
            for substr in location_str.split(' '):
                location_set.add(substr)
    return location_set

# Return all time-related (token) string
def find_all_time(sentence):
    time_set = set()
    for ent in sentence.ents:
        if ent.label_ == 'DATE' or ent.label_ == 'TIME':
            for substr in ent.text.split(' '):
                time_set.add(substr)
    return time_set

def find_root_in_sentence(sentence):
    result = ''
    for token in sentence:
        if token.dep_ == 'ROOT':
            result = token
    return result

# Return the proper verb to be inserted to ask a question.
def convert_root_to_proper_verb(root, sentence):
    # past participate, "was completed" -> was
    if root.tag_ == 'VBN':
        for token in sentence:
            if token.pos_ == 'AUX':
                return token.text.lower()

    # aux verb -> aux
    if root.pos_ == 'AUX':
        return root.text.lower()

    if root.tag_ == 'VBD' or root.tag_ == 'VBN':
        return 'did'
    elif root.tag_ == 'VBZ':
        return 'does'
    elif root.tag_ == 'VBP':
        return 'do'

    return ''


def get_verb_insertion_index(sentence):
    verb_index = 0
    subject_index = 0

    for i, token in enumerate(sentence):
        if token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass':
            subject_index = i
            break

    for i, token in enumerate(sentence):
        if i < subject_index and (token.text == ',' or token.text == '\n'):
            verb_index = i + 1

    return verb_index


# Convert result, which is an array of Strings, to the final output string
def construct_string_old(result):
    replace_abbrevation(result)
    output_string = ''

    skip_hyphen = False

    for i, word in enumerate(result):
        if len(word) == 0:
            continue

        if i == 0:
            word = capitalize_first_character(word)

        # No preceding space
        if word == "'s":
            output_string += word
            continue

        if word != ',' and word != '?' and i != 0 and word != "'" and word != '.' and word != '-':
            if not skip_hyphen:
                output_string += ' '
            skip_hyphen = False

        if word == '-':
            skip_hyphen = True

        output_string += word
    return output_string


def construct_string(result):
    replace_abbrevation(result)

    output_string = ''

    double_quotation_count = 0

    prev_word = ''

    for i, word in enumerate(result):
        word = word.strip()

        if i == 0:
            word = capitalize_first_character(word)

        if len(word) == 0:
            continue

        need_space = True

        if i == 0 or word == ')' or word == ']' or word == '}'\
                or (word == '"' and double_quotation_count == 1) \
                or prev_word in no_suffix_space_words \
                or word in no_prefix_space_word:
            need_space = False

        if prev_word == '"' and double_quotation_count == 0:
            need_space = True

        if i == len(result) - 1 and not word.isalpha():
            need_space = False

        if word == "'" and prev_word.endswith('s'):
            need_space = False

        if word == '':
            need_space = False

        if need_space:
            output_string += ' '

        output_string += word

        prev_word = word

        if word == '"':
            if double_quotation_count == 1:
                double_quotation_count = 0
            else:
                double_quotation_count = 1

    return output_string

def get_sentence_at_index(document, index):
    for i, sentence in enumerate(document.sents):
        if i == index:
            return str(sentence)


def contains_negate(sentence):
    for token in sentence:
        if token.dep_ == 'neg':
            return True
    return False


def get_answer_consider_negate(answer, num_negate):
    if answer == 'Yes.':
        if num_negate % 2 == 0:
            return 'Yes.'
        else:
            return 'No.'
    else:
        if num_negate % 2 == 0:
            return 'No.'
        else:
            return 'Yes.'


# Return the location token string in an array. Empty means no location found after root.
def get_location_after_root(sentence, where_question_root):
    location_ner_string_set = find_all_location(sentence)

    encountered_root = False

    result = []
    for token in sentence:
        if token.lemma_ == where_question_root.lemma_:
            encountered_root = True

        if encountered_root:
            if (token.dep_ == 'pobj' or token.dep_ == 'compound') and token.text in location_ner_string_set:
                result.append(token.text)

        if encountered_root and token.dep_ == 'nsubj':
            break

    return result

# Return the time token string in an array. Empty means no time found after root.
def get_time_after_root(sentence, when_question_root):
    time_ner_string_set = find_all_time(sentence)

    encountered_root = False

    result = []
    for token in sentence:
        if token.lemma_ == when_question_root.lemma_:
            encountered_root = True

        if encountered_root and token.text in time_ner_string_set:
            result.append(token.text)

        if encountered_root and (token.dep_ == 'nsubj' or token.dep_ == 'cc'):
            break

    return result

# Return the time token string in an array. Empty means no time found after root.
def get_attr_after_root(sentence, who_aux_question_root):

    encountered_root = False
    encountered_attr = False

    result = []
    for token in sentence:
        if token.lemma_ == who_aux_question_root.lemma_:
            encountered_root = True

        if encountered_root:
            if token.dep_ == 'attr':
                encountered_attr = True

            if token.dep_ == 'attr' or token.dep_ == 'compound':
                result.append(token.text)
            elif encountered_attr:
                result.append(token.text)

        if encountered_root and (token.dep_ == 'conj' or token.dep_ == 'cc'):
            break

    return result

# get prefix for who/what auxilary question
def get_who_aux_question_prefix(question):
    question_root = find_root_in_sentence(question)
    result = []

    encountered_root = False
    for token in question:
        if token.text == question_root.text:
            encountered_root = True

        if encountered_root and (token.dep_ == 'nsubj' or token.dep_ == 'attr' or token.dep_ == 'compound'):
            result.append(token.text)
            if token.dep == 'attr' or token.dep_ == 'nsubj':
                break

    result.append(question_root.text)

    return result


def find_determiner_in_sentence(sentence):
    encountered_root = False
    for token in sentence:
        if token.dep_ == 'ROOT':
            encountered_root = True

        if encountered_root and token.dep_ == 'det' and token.pos_ == 'DET' \
                and (token.text == 'a' or token.text == 'an' or token.text == 'the'):
            return token.text
    return ''


def examine_sentence_component(sentence):
    print('\n---------')
    print('Sentence: ' + str(sentence))
    for token in sentence:
        print(token.text, token.dep_, token.pos_, token.head.text, [child for child in token.children])
    print('---------\n')


def replace_abbrevation(string_array):
    for i, word in enumerate(string_array):
        if word == "n't":
            string_array[i] = 'not'
    return string_array


def get_next_token_after_root(sentence):
    encountered_root = False
    for token in sentence:
        if token.dep_ == 'ROOT':
            encountered_root = True
            continue
        if encountered_root:
            return token

    return None


def find_name_words_in_who_question(question):
    result = []
    for i, token in enumerate(question):
        if token.ent_type_ == 'PERSON':
            result.append(token.text)
    return result


def get_all_person_ner(sentence):
    result = set()
    for ent in sentence.ents:
        if ent.label_ == 'PERSON' or ent.label_ == 'ORG':
            result.add(ent.text)
    return result


def get_all_non_person_ner(sentence):
    result = []
    for ent in sentence.ents:
        if ent.label_ != 'PERSON' or ent.label_ == 'ORG':
            result.append(ent.text)
    return result


def combine_missing_ner(missing_ner):
    result = ''
    for i, ner in enumerate(missing_ner):
        if i < len(missing_ner) - 2:
            result = result + ner + ', '
        elif i == len(missing_ner) - 2:
            result = result + ner + ' and '
        elif i == len(missing_ner) - 1:
            result = result + ner

    return result


# Get all the subject, including compound (before ROOT) in a sentence
def get_subject_in_sentence(sentence):
    result = []

    encountered_nsubj = False
    for token in sentence:
        if token.dep_ == 'nsubj':
            encountered_nsubj = True
        if token.dep_ == 'punct' and not encountered_nsubj:
            result = []
        elif token.dep_ == 'ROOT':
            return result
        else:
            result.append(token.text)

    return result


def read_question_file(questionFileLocation):
    file1 = open(questionFileLocation, 'r', encoding='utf-8')
    lines = file1.readlines()
    content = [x.strip() for x in lines]
    return content


def get_nsubj(sentence):
    for token in sentence:
        if (token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass') and token.pos_ != 'PRON':
            return token.lemma_

    return ''


# -------------- Post dry run ----------------
# returns the token index of target_token in sentence
def get_token_index_in_sentence(sentence, target_token):
    for i, token in enumerate(sentence):
        if token.text == target_token.text and token.dep_ == target_token.dep_ and token.pos_ == target_token.pos_:
            return i

    return -1


# Get the first child of root_token that appears after root in sentence
def get_first_child_after_root(sentence, root_token):
    root_token_index = get_token_index_in_sentence(sentence, root_token)

    for child in root_token.children:
        curr_child_token_index = get_token_index_in_sentence(sentence, child)
        if curr_child_token_index > root_token_index:
            return child

    return None


# Recursively get all of token's children in sentence before token (exclusive)
def get_children_before_token(sentence, token, result):
    token_index = get_token_index_in_sentence(sentence, token)
    for child in token.children:
        child_index = get_token_index_in_sentence(sentence, child)
        if child_index >= token_index:
            continue

        if len(list(child.children)) != 0:
            get_children_before_token(sentence, child, result)

        result.append(child.text)


def get_original_sentence_index(document, matched_sentence):
    for i, sentence in enumerate(document.sents):
        if matched_sentence.lower() in str(sentence).lower():
            return i

    return 0


def get_location_ent_str(sentence):
    location_set = set()
    for ent in sentence.ents:
        if ent.label_ == 'GPE' or ent.label_ == 'LOC':
            location_set.add(str(ent.text))
    return location_set


def get_average_length_from_all_questions(all_question_list):
    question_num = len(all_question_list)
    length_sum = 0
    for question_tuple in all_question_list:
        length_sum += question_tuple[0]

    return math.ceil(length_sum / question_num)
