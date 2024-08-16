#!/usr/bin/env python3
import queue

import spacy
from preprocessor import *
import sys
from pysbd.utils import PySBDFactory
from coreference_resolver import *
from grammar_checker import *
import neuralcoref


have_variations = ['have', 'has', 'had']

class questionGeneration:

    def __init__(self):
        pass

    def generate_binary_question(self, sentence):

        result = []
        root = find_root_in_sentence(sentence)

        verb_to_be_inserted = convert_root_to_proper_verb(root, sentence)
        verb_inserted = False
        skipped_verb = False
        verb_insertion_index = get_verb_insertion_index(sentence)

        ner_set = find_all_ner_need_capitalized(sentence)

        first_valid_index = get_first_valid_index(sentence)

        for i, token in enumerate(sentence):
            if i < first_valid_index:
                continue

            if not verb_inserted and len(result) == verb_insertion_index:
                if len(result) == 0:
                    verb_to_be_inserted = capitalize_first_character(verb_to_be_inserted)
                result.append(verb_to_be_inserted)
                verb_inserted = True

            if i == 0:
                # check for first word, if it is not an NER, no longer need to capitalize first character
                if token.text in ner_set or token.text == 'I':
                    result.append(token.text)
                else:
                    if len(result) != 0:
                        result.append(token.text.lower())
                    else:
                        result.append(token.text)
                continue

            #if token.text != root.text or skipped_verb:
            if token.dep_ != 'ROOT':
                if token.pos_ == 'AUX' and root.tag_ == 'VBN':
                    continue
                result.append(token.text)
            else:
                skipped_verb = True
                if not is_be_verb(token.text):
                    # don't change tense if it is VPN
                    if root.tag_ != 'VBN':
                        result.append(token.lemma_)
                    else:
                        result.append(token.text)

        if len(result) > 0:
            result[-1] = '?'
        return construct_string(result)

    def generate_where_question(self, sentence):
        location_set = find_all_location(sentence)

        if len(location_set) == 0:
            return ''

        result = []
        root = find_root_in_sentence(sentence)

        verb_to_be_inserted = convert_root_to_proper_verb(root, sentence)
        verb_inserted = False
        verb_insertion_index = get_verb_insertion_index(sentence)

        ner_set = find_all_ner_need_capitalized(sentence)

        first_valid_index = get_first_valid_index(sentence)

        for i, token in enumerate(sentence):
            if i < first_valid_index:
                continue

            if token.text in location_set:
                continue

            if not verb_inserted and len(result) == verb_insertion_index:
                if len(result) == 0:
                    result.append('Where')
                else:
                    result.append('where')
                result.append(verb_to_be_inserted)
                verb_inserted = True

            if i == 0:
                # check for first word, if it is not an NER, no longer need to capitalize first character
                if token.text in ner_set or token.text == 'I':
                    result.append(token.text)
                else:
                    if len(result) != 0:
                        result.append(token.text.lower())
                    else:
                        result.append(token.text)
                continue

            if token.dep_ != 'ROOT':
                # avoid duplicated is for sentence like: XX is in XX
                if token.pos_ == 'AUX' and root.tag_ == 'VBN':
                    continue
                result.append(token.text)
            else:
                if token.pos_ != 'AUX':
                    # don't change tense if it is VPN
                    if root.tag_ != 'VBN':
                        result.append(token.lemma_)
                    else:
                        result.append(token.text)
        if len(result) > 0:
            result[-1] = '?'
        return construct_string(result)

    def generate_what_question(self, sentence):
        non_person_ent_set = find_all_non_person_subj(sentence)
        has_auxiliary = contains_aux_for_what(sentence)

        if len(non_person_ent_set) == 0 or not has_auxiliary:
            return ''

        result = []
        root = find_root_in_sentence(sentence)

        verb_to_be_inserted = convert_root_to_proper_verb(root, sentence)
        verb_inserted = False
        verb_insertion_index = get_verb_insertion_index(sentence)

        first_valid_index = get_first_valid_index(sentence)

        need_capitalize = find_all_ner_need_capitalized(sentence)

        for i, token in enumerate(sentence):
            if i < first_valid_index:
                continue

            if not verb_inserted and i == verb_insertion_index:
                if len(result) == 0:
                    result.append('What')
                else:
                    result.append('what')
                result.append(verb_to_be_inserted)
                break

        num_subject_remaining = len(non_person_ent_set)
        for subject in non_person_ent_set:
            if subject in need_capitalize:
                result.append(subject)
            else:
                result.append(subject.lower())
            num_subject_remaining -= 1
            if num_subject_remaining >= 2:
                result.append(',')
            elif num_subject_remaining == 1:
                result.append('and')

        if len(result) > 0:
            result.append('?')
        return construct_string(result)

    # Strategy: a sentence need to have >= 1 person subject(s) and auxiliary to be able to generate a who question.
    def generate_who_question_aux(self, sentence):
        person_ent_set = find_all_person_subj(sentence)
        has_auxiliary = contains_aux_for_what(sentence)

        if len(person_ent_set) == 0 or not has_auxiliary:
            return ''

        result = []
        root = find_root_in_sentence(sentence)
        verb_to_be_inserted = convert_root_to_proper_verb(root, sentence)
        verb_inserted = False
        verb_insertion_index = get_verb_insertion_index(sentence)

        first_valid_index = get_first_valid_index(sentence)

        for i, token in enumerate(sentence):
            if i < first_valid_index:
                continue

            if not verb_inserted and i == verb_insertion_index:
                if len(result) == 0:
                    result.append('Who')
                else:
                    result.append('who')
                result.append(verb_to_be_inserted)
                break

        num_person_remaining = len(person_ent_set)
        for person in person_ent_set:
            result.append(person)
            num_person_remaining -= 1
            if num_person_remaining >= 2:
                result.append(',')
            elif num_person_remaining == 1:
                result.append('and')

        if len(result) > 0:
            result.append('?')
        return construct_string(result)

    def generate_who_question_verb(self, sentence):
        person_ent_set = find_all_person_subj(sentence)
        has_auxiliary = contains_aux_for_what(sentence)

        if has_auxiliary:
            return ''

        root = find_root_in_sentence(sentence)

        if len(person_ent_set) == 0 or len(root.text) == 0:
            return ''

        result = []

        verb_inserted = False
        verb_insertion_index = get_verb_insertion_index(sentence)

        first_valid_index = get_first_valid_index(sentence)

        for i, token in enumerate(sentence):
            if i < first_valid_index:
                continue

            if not verb_inserted and i == verb_insertion_index:
                if len(result) == 0:
                    result.append('Who')
                else:
                    result.append('who')

            is_person_name = False
            for person_ent_str in person_ent_set:
                if token.text in person_ent_str:
                    is_person_name = True

            if not is_person_name and token.text != '.':
                result.append(token.text)

        if len(result) > 0:
            result.append('?')
        return construct_string(result)

    def generate_when_question(self, sentence):
        time_ent_set = find_all_time_ent(sentence)

        if len(time_ent_set) == 0:
            return ''

        time_prep_str = find_prep_before_time_ent(sentence, time_ent_set)
        root = find_root_in_sentence(sentence)
        verb_to_be_inserted = convert_root_to_proper_verb(root, sentence)

        result = []

        when_inserted = False
        verb_insertion_index = get_verb_insertion_index(sentence)

        first_valid_index = get_first_valid_index(sentence)

        for i, token in enumerate(sentence):
            if i < first_valid_index:
                continue

            if not when_inserted and i == verb_insertion_index:
                if len(result) == 0:
                    result.append('When')
                else:
                    result.append('when')
                result.append(verb_to_be_inserted)
                when_inserted = True

            if not when_inserted:
                continue

            if when_inserted and token.text == time_prep_str:
                continue

            is_time = False
            for time_str in time_ent_set:
                if token.text in time_str:
                    is_time = True

            if not is_time and token.text != '.':
                if token.text == verb_to_be_inserted:
                    if is_be_verb(token.text):
                        continue
                    else:
                        result.append(token.lemma_)
                else:
                    if token.pos_ == 'VERB' and token.text == root.text:
                        if token.tag_ == 'VBN':
                            result.append(token.text)
                        else:
                            result.append(token.lemma_)
                    else:
                        result.append(token.text)

        if len(result) > 0:
            result.append('?')
        return construct_string(result)


if __name__ == '__main__':
    DEBUG = True
    file_name = sys.argv[1]
    num_questions = int(sys.argv[2])

    # python -m spacy download en_core_web_lg
    sentence_tokenizer = spacy.blank('en')
    sentence_tokenizer.add_pipe(PySBDFactory(sentence_tokenizer))
    preprocessed_content = preprocess_input(sentence_tokenizer, file_name) # Return value is a simple String

    pysbd_docuemnt = sentence_tokenizer(preprocessed_content)
    valid_sentnces_set = set()
    for sentnece in pysbd_docuemnt.sents:
        valid_sentnces_set.add(str(sentnece))

    small_nlp = spacy.load('en_core_web_sm')
    large_nlp = spacy.load('en_core_web_lg')

    # --------- Caution -----------
    # document will be used by coref_resolver, to insure neuralcoref (coref_resolver) and the document
    # have parallel sentence index, must use the same nlp language model in the below 2 lines
    neuralcoref.add_to_pipe(small_nlp)        # Need to add neuralcoref to pipeline before generating documents and sentences
    document = small_nlp(preprocessed_content)
    # -----------------------------

    questionGenerationUnit = questionGeneration()

    all_questions = []

    coref_resolver = CoreferenceResolver(small_nlp, document)

    grammer_checker = GrammarChecker(large_nlp)

    for i, sentence in enumerate(document.sents):
        if len(str(sentence)) <= 1 or str(sentence) not in valid_sentnces_set:
            continue

        # if 'wolves almost always have amber or light colored eyes' in str(sentence):
        #     print('Stop')

        # Re-process as sentence comes from document (small_nlp)
        processed_sentence = small_nlp(str(sentence))

        binary_question = questionGenerationUnit.generate_binary_question(processed_sentence) # Good

        resolved_coref_binary_question = coref_resolver.resolve_binary_question_coreference(binary_question, i)

        # if DEBUG:
        #     print('Original --- ' + str(sentence))

        # question_tuple (length (int), grammar_checker (boolean), coref_updated (boolean), question_str (string))
        if len(binary_question) > 0:
            # if DEBUG:
            #     print('Binary (Before) --- ' + binary_question)
            #     print('Binary (After)  --- ' + resolved_coref_binary_question)

            if resolved_coref_binary_question.startswith(' '):
                continue

            question_length = len(resolved_coref_binary_question.split())
            contains_grammar_error = grammer_checker.contains_grammar_error(resolved_coref_binary_question)
            coref_updated = binary_question != resolved_coref_binary_question
            all_questions.append((question_length, contains_grammar_error, coref_updated, resolved_coref_binary_question))

        # print('*** Generated ' + str(len(all_questions)) + ' questions.***\n')
        # print('Valid questions = ' + str(len(all_questions)))

    average_length = get_average_length_from_all_questions(all_questions) * 0.8
    # print('average_length = ' + str(average_length))

    priority_queue = queue.PriorityQueue()

    for question_tuple in all_questions:
        curr_list = list(question_tuple)
        curr_list[0] = abs(average_length - curr_list[0])
        new_tuple = tuple(curr_list)
        priority_queue.put(new_tuple)

    final_result = []
    l1 = []     # Grammar correct, but has been updated by neuralcoref
    l2 = []     # Grammar incorrect

    while not priority_queue.empty():
        curr_question_tuple = priority_queue.get()
        curr_question_str = curr_question_tuple[3]
        first_word = curr_question_str.split()[0].lower()

        if curr_question_tuple[1] \
                or first_word in have_variations \
                or 'although' in curr_question_str.lower():          # Contains grammar error
            l2.append(curr_question_tuple)
        elif curr_question_tuple[2] or '(' in curr_question_str or '[' in curr_question_str:            # Coref updated
            l1.append(curr_question_tuple)
        else:                                   # Grammar correct and not coref updated
            final_result.append(curr_question_tuple)

    final_result.extend(l1)
    final_result.extend(l2)

    for i in range(min(num_questions, len(final_result))):
        print(final_result[i][3])
