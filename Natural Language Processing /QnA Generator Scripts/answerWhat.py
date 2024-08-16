from utils import *

be_aux_list = ['am', 'is', 'are', 'was', 'were']


class AnswerWhat:

	def __init__(self, nlp):
		self.nlp = nlp

	def answerWhatQuestion(self, question, resolved_coref_sentence):
		processed_question = self.nlp(question)
		processed_text = self.nlp(resolved_coref_sentence)

		# examine_sentence_component(processed_question)
		# examine_sentence_component(processed_text)
		root = find_root_in_sentence(processed_question)

		question_nsubj = get_nsubj(processed_question).lower()
		text_nsubj = get_nsubj(processed_text).lower()

		if str(root) in be_aux_list:
			if question_nsubj == text_nsubj:
				return self.answerWhatAttrQuestion(question, resolved_coref_sentence)
			else:
				return construct_string(get_subject_in_sentence(processed_text))
		else:
			return resolved_coref_sentence

	def answerWhatAttrQuestion(self, question, resolved_coref_sentence):
		processed_question = self.nlp(question)
		processed_text = self.nlp(resolved_coref_sentence)

		question_root = find_root_in_sentence(processed_question)

		answer = get_attr_after_root(processed_text, question_root)
		# prefix = get_who_aux_question_prefix(processed_question)
		# det = find_determiner_in_sentence(processed_text)
		# prefix.append(det)
		# prefix.extend(answer)
		if len(answer) != 0:
			# return construct_string(prefix)
			return construct_string(answer)
		else:
			return resolved_coref_sentence

