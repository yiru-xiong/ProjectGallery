import numpy as np
from fuzzywuzzy import fuzz

class ExtractRelevantSentences:
    def __init__(self, mode, large_nlp):
        # if mode = 1: find the top n sentences with the highest similarity scores
        # similarity score is defined as numpy.dot(self.vector, other.vector) / (self_norm * other_norm))
        self.mode = mode
        self.nlp = large_nlp
        self.similarity_threshold = 0.9

    def text_lemmatization(self, sentence):
        new_sentence = []
        for token in self.nlp(sentence):
            new_sentence.append(token.lemma_)
        return ' '.join(new_sentence)

    def remove_punct_stopwords(self, sentence):
        cleaned_sentence = [token.text for token in sentence if not token.is_stop]
        return ' '.join(cleaned_sentence)

    def remove_first_word(self, sentence):
        wh_question_list = ["who", "where", "when", "what", "why"]
        if sentence.split(' ')[0].lower() in wh_question_list:
           sentence = sentence.split(' ', 1)[1]
        return sentence
        
    # compute the similarity scores and get index of top n
    def get_similarity_scores(self, full_text, question_list, n):
        context_dict = {}  # question_idx -> list of top original sentence index
        for q_idx, q_str in enumerate(question_list):
            q = self.remove_first_word(q_str)
            question = self.nlp(q)
            cleaned_question = self.nlp(self.remove_punct_stopwords(question))
            similarity_scores = []
            # use only Spacy Similarity
            if self.mode == 1:
                for sentence_idx, sentence in enumerate(self.nlp(full_text).sents):
                    cleaned_sentence = self.nlp(self.remove_punct_stopwords(sentence))
                    similarity_scores.append(cleaned_question.similarity(cleaned_sentence))
                top_idxes = np.argsort(similarity_scores)[::-1][:n]

            # use FuzzWuzzy 
            elif self.mode == 2:
                for sentence_idx, sentence in enumerate(self.nlp(full_text).sents):
                    cleaned_sentence = self.nlp(self.remove_punct_stopwords(sentence))
                    similarity_scores.append(fuzz.partial_ratio(str(cleaned_question), str(cleaned_sentence)))
                top_idxes = np.argsort(similarity_scores)[::-1][:n]
            
            # combination of Spacy Similarity + FuzzyWuzzy
            elif self.mode == 3:
                similarity_scores_check1 = []
                similarity_scores_check2 = []
                for sentence_idx, sentence in enumerate(self.nlp(full_text).sents):

                    cleaned_sentence = self.nlp(self.remove_punct_stopwords(sentence))
                    similarity_scores_check1.append(cleaned_question.similarity(cleaned_sentence))
                    similarity_scores_check2.append(fuzz.partial_ratio(str(cleaned_question), str(cleaned_sentence)))
                top_idxes_check1 = np.argsort(similarity_scores_check1)[::-1][:1]

                spacy_similarity = similarity_scores_check1[top_idxes_check1[0]]

                top_idxes_check2 = np.argsort(similarity_scores_check2)[::-1][:1]
                fw_similarity = similarity_scores_check2[top_idxes_check2[0]]

                # print('SpaCy --- ' + str(spacy_similarity))
                # print('FW --- ' + str(fw_similarity))

                # if the max similarity score is less than the threshold, factor in FuzzyWuzzy scores
                if spacy_similarity < self.similarity_threshold:
                   check1_top_5 =  np.argsort(similarity_scores_check1)[::-1][:5]
                   top_idxes_check2 =   np.argsort(similarity_scores_check2)[::-1][:1]

                   # use FuzzyWuzzy result if its max score also appears in the top 3 highest similarity score list, or it gt threshold*100
                   if (top_idxes_check2[0] in list(check1_top_5)) or (fw_similarity > spacy_similarity * 110) or fw_similarity >= 95:
                       top_idxes = top_idxes_check2
                   else:
                       top_idxes = top_idxes_check1
                else:
                    top_idxes = top_idxes_check1

            context_dict[q_idx] = list(enumerate(top_idxes))

        return context_dict

    # extract the top n sentences based on indexes
    def get_top_n_sentences(self, full_text, question_list, n):
        question_context_dic = {}  # question_idx -> list of most likely n sentences (from most likely to least likely)
        question_idx_dic = self.get_similarity_scores(full_text, question_list, n)
        for question_idx in question_idx_dic.keys():
            top_n_sentences = []  # tuple(rank, sentence_index, sentence) in original text
            for sentence_idx, sentence in enumerate(self.nlp(full_text).sents):
                if sentence_idx in [idx[1] for idx in question_idx_dic[question_idx]]:
                    rank = next((order_num for order_num, sent_idx in question_idx_dic[question_idx] if
                                 sent_idx == sentence_idx), None)
                    top_n_sentences.append((rank, sentence_idx, sentence.text))
                top_n_sentences.sort(key=lambda x: x[0])
            extracted_sentence_with_idx = [[element[1], str(element[2])] for element in top_n_sentences]
            question_context_dic[question_idx] = extracted_sentence_with_idx
        return question_context_dic
        # if n=2, output -> {question_num:[[sentence_idx_top1, sentence1],[sentence_idx_top2, sentence2]]}
