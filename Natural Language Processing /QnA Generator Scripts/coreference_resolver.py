from utils import *

no_need_to_replace = ['who', 'that', 'this', 'whose', 'those']

# used by coreference_resolver.binary_resolve_coreference, return the subject after first aux.
# Parameter sentence should have been wrapped in nlp before. Return co-reference in String format if found
def get_coreference_after_first_aux(sentence):
	encountered_first_aux = False

	for token in sentence:
		if token.pos_ == 'AUX':
			encountered_first_aux = True
			continue

		# Requirement: appear after first aux, has to be subject or poss, and pos != 'PROPN', which is the actual name
		if encountered_first_aux:
			if (token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass' or token.dep_ == 'poss') and token.pos_ != 'PROPN' and token.pos_ != 'NOUN':
				return token.text
			else:
				continue

	return ''  # No co-reference found


def get_proper_main_label(main_label, coreference_str):
	coreference_str = coreference_str.lower()

	if coreference_str == 'his' or coreference_str == 'her' or coreference_str == 'its':
		if main_label.endswith('s'):
			main_label += "'"
		else:
			main_label += "'s"

	if coreference_str == 'their':
		if main_label.endswith('s'):
			main_label += "'"
		else:
			main_label += "'s"

	return main_label


class CoreferenceResolver:
	# nlp should have included neuralcoref pipeline, document should have been wrapped by nlp
	def __init__(self, nlp, docuemnt):
		self.nlp = nlp
		self.document = docuemnt

	# Check if generated binary question contains co-reference.
	# The observation is that the first word that has pos = 'AUX' in the question is typically something like 'did',
	# 'does' and 'was' etc. If there's any co-reference, it's typically the word immediately after 'AUX' and has
	# dep = 'nsubj', 'nsubjpass' (passive) and poss.
	def resolve_binary_question_coreference(self, generated_question_str, sentence_index):
		generated_question = self.nlp(generated_question_str)

		coreference_str = get_coreference_after_first_aux(generated_question)  # to be replaced by main label

		if len(coreference_str) == 0:
			return generated_question_str  # no co-reference founded

		result = []

		for token in generated_question:
			if token.text.lower() in no_need_to_replace or token.text != coreference_str:
				result.append(token.text)
			else:
				main_label = self.get_coref_main_label(sentence_index, coreference_str)
				result.append(main_label)

		return construct_string(result)

	def resolve_original_sentence_coreference(self, sentence_index):
		original_sentence = list(self.document.sents)[sentence_index]

		result = []

		for token in original_sentence:
			# Find 'he, him, her, them, they' etc
			if (token.dep_ == 'nsubj' or token.dep_ == 'dobj' or token.dep_ == 'nsubjpass' or token.dep_ == 'dative') \
				and token.pos_ == 'PRON' \
				and token.text.lower() != 'who'\
				and token.text.lower() != 'that'\
				and token.text.lower() != 'this':
				if token._.in_coref:
					main_label = str(token._.coref_clusters[0].main)
					result.append(main_label)
					continue

			# Find 'his, her, their' etc
			if token.dep_ == 'poss' and token.pos_ == 'DET':
				if token._.in_coref:
					main_label = str(token._.coref_clusters[0].main)
					result.append(get_proper_main_label(main_label, token.text))
					continue

			result.append(token.text)

		return construct_string(result)

	# Get the coref main label of the coreference_str from the corresponding original sentence in document
	# sentence_index with regard the nlp model applied to self.document
	def get_coref_main_label(self, sentence_index, coreference_str):

		original_sentence = list(self.document.sents)[sentence_index]
		coreference_str = coreference_str.lower()

		for token in original_sentence:
			if token.text.lower() == coreference_str and token._.in_coref:
				main_label = str(token._.coref_clusters[0].main)
				return get_proper_main_label(main_label, coreference_str)


		return ''
