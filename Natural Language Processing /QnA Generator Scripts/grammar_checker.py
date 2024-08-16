from spacy.matcher import Matcher


class GrammarChecker:
    def __init__(self, large_nlp):
        # if mode = 1: find the top n sentences with the highest similarity scores
        # similarity score is defined as numpy.dot(self.vector, other.vector) / (self_norm * other_norm))
        self.nlp = large_nlp

        # PROP + this/that
        prop_list = ["she", "he", "it", "they", "this", "that", "these", "those"]
        # question_with_base_verb = ["do","does", "did", "would", "may", "might", "should", "will", "don't", "doesn't", "didn't", "wouldn't", "shouldn't"]
        # aux_must_follow_by_np = ["do", "does", "did"]
        aux_with_non_base_verb = ["is","are","am","was","were","has","have","had"]
        disallowed_second_aux = ["is", "are", "am", "do", "does", "was", "were","did"]

        # bad grammar patterns in generated questions
        bad_ask_grammar_patterns = [
        # Pattern 1: consecutive aux
        [{'POS':'AUX'}, {'POS':'AUX'}],
        # pattern 2: aux+existence of pron(she/he/it/there)
        [{'POS':'AUX'}, {'LOWER': {'IN':prop_list}}],
        # pattern 3: md + existence of pron
        [{'TAG':'MD'}, {'LOWER': {'IN':prop_list}}],
        # pattern 4: is/are/has/have  + optional noun phrase +  base-form verb
        [{'LOWER':{'IN':aux_with_non_base_verb}}, {'POS': 'DET', 'OP':'?'}, {'POS': 'ADV', 'OP':'*'}, {'POS': 'ADJ', 'OP':'*'}, {'POS': {'IN':['PROPN', 'NOUN']}, 'OP':'*'}, {'TAG': 'VB'}],
        # pattern 5: aux + optional noun phrase + disallowed_aux
        [{'POS':'AUX'}, {'POS': 'DET', 'OP':'?'}, {'POS': 'ADV', 'OP':'*'}, {'POS': 'ADJ', 'OP':'*'}, {'POS': {'IN':['PROPN', 'NOUN']}, 'OP':'*'}, {'LOWER':{'IN':disallowed_second_aux}}],
        # pattern 6: pron/propn/noun + md
        [{'POS': {'IN':['PROPN','PRON','NOUN']}}, {'TAG':'MD'}],
        # pattern 7: and/then/but + pron + non base-form verb
        [{'LOWER':{'IN':['and', 'and then', 'but']}}, {'POS':'PRON'}, {'TAG':'VB', 'OP':'!'}]
        # pattern 8: tense mismatch - verb should be in base form but not
        #[{'LOWER':{'IN':question_with_base_verb}}, {'POS': 'DET', 'OP':'?'}, {'POS': 'ADJ', 'OP':'*'}, {'POS': {'IN':['PROPN', 'NOUN']}, 'OP':'?'},{'TAG': 'VB', 'OP': '!'}],
        # pattern 9: aux(do/does/did) + not a noun phrase
        #[{'LOWER':{'IN':aux_must_follow_by_np}}, {'POS': 'DET', 'OP':'?'},{'POS': 'ADV', 'OP':'*'}, {'POS': 'ADJ', 'OP':'*'}, {'POS': {'IN':['PROPN','PRON','NOUN']}, 'OP':'!'}]
        ]
        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add('Bad Grammar Pattern-1', None, list(bad_ask_grammar_patterns[0]))
        self.matcher.add('Bad Grammar Pattern-2', None, list(bad_ask_grammar_patterns[1]))
        self.matcher.add('Bad Grammar Pattern-3', None, list(bad_ask_grammar_patterns[2]))
        self.matcher.add('Bad Grammar Pattern-4', None, list(bad_ask_grammar_patterns[3]))
        self.matcher.add('Bad Grammar Pattern-5', None, list(bad_ask_grammar_patterns[4]))
        self.matcher.add('Bad Grammar Pattern-6', None, list(bad_ask_grammar_patterns[5]))
        self.matcher.add('Bad Grammar Pattern-7', None, list(bad_ask_grammar_patterns[6]))
        #self.matcher.add('Bad Grammar Pattern-8', None, list(bad_ask_grammar_patterns[7]))
        #self.matcher.add('Bad Grammar Pattern-9', None, list(bad_ask_grammar_patterns[8]))

    def contains_grammar_error(self, question):
        question = self.nlp(question)
        matches = self.matcher(question)
        if len(matches) != 0:
           return True # spot a grammar error
        else:
           return False # did not spot a grammar error
