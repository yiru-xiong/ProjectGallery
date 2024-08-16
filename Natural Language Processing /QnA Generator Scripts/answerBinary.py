from utils import *

binary_question_list = ['do', 'does', 'did', 'are', 'is', 'am',
                        'was', 'were', 'have', 'has', 'had',
                        'should', 'shall', 'may', 'might', 'must',
                        'can', 'could', 'need', 'will', 'would']


class AnswerBinary:

    def __init__(self, nlp):
        self.nlp = nlp

    def answerBinaryQuestion(self, question, resolved_coreference_sentence):

        processed_question = self.nlp(question)
        processed_text = self.nlp(resolved_coreference_sentence)

        num_negate = 0
        if contains_negate(processed_question):
            num_negate += 1
        if contains_negate(processed_text):
            num_negate += 1


        text_token_str_set = set()

        for token in processed_text:
            text_token_str_set.add(str(token.lemma_))
            text_token_str_set.add(token.text)

        token_overlap = True
        for token in processed_question:
            if token.lemma_ in binary_question_list or token.pos_ == 'PUNCT' or token.pos_ == 'PRON':
                continue

            if token.lemma_ not in text_token_str_set and token.text not in text_token_str_set:
                token_overlap = False
                break

        if token_overlap:
            return get_answer_consider_negate('Yes.', num_negate)


        nouns_in_question = get_noun_lemma_from_sentence(processed_question)
        nouns_in_text = get_noun_lemma_from_sentence(processed_text)
        names_ent_in_question = find_names_in_question(processed_question)

        for noun in nouns_in_question:
            if noun not in nouns_in_text:
                return get_answer_consider_negate('No.', num_negate)

        for name in names_ent_in_question:
            found_name = False
            for word in name:
                if word in processed_text.text:
                    found_name = True
            if not found_name:
                return get_answer_consider_negate('No.', num_negate)

        return get_answer_consider_negate('Yes.', num_negate)
