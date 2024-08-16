###########################################################
####### Sentiment Classification using Naive Bayes  #######
###########################################################
# author: Yiru Xiong

# import libraries
import sys
import io
import numpy as np
import math
from collections import defaultdict
import re

class NaiveBayes(object):

    def __init__(self, tokenizationMethod="unigram", trainingData, trainingLabel):
        """
        Initializes a NaiveBayes object and the naive bayes classifier.
        param trainingData: full text from training file as a single large string
        """

        # set of all claases
        classes = set(trainingLabel)
        vocabulary = set()
        
        # create vocabulary based on chosen tokenization method unigram or bigram 
        if tokenizationMethod == "bigram":
            # counts of bi-grams:
            for s_input in trainingData:
                words = s_input.split(" ") 
                for i in range(len(words)-1):
                    bigram=words[i]+' '+words[i+1]
                    vocabulary.add(bigram)  
        else:
            counts of all unique words 
            for s_input in trainingData:
                for word in s_input.split(" "):
                    word = re.sub('[^A-Za-z0-9_]+', '',word)
                    if word != '':
                    vocabulary.add(word)

        vocab_size = len(vocabulary)
        #print(vocab_size)
        
        dic = {}
        for x, y in zip(trainingData, trainingLabel):
            if y not in dic:
                dic[y]= []
            dic[y].append(x)
        
        # create a dictionary with key=class label and value = sentences in the class
        #dic = {}
        counts = {}
        for c in list(dic.keys()):
            sentence_list = dic[c]
            if c not in counts:
                counts[c]={}
            #counts[c] = defaultdict(int)
            for sen in sentence_list:
                #sen = re.sub('[^A-Za-z0-9_ ]+', '',sen)
                ws = sen.split(" ")
                for w in ws:
                    #counts[c][w]+= 1
                    if w not in counts[c]:
                        counts[c][w]=1
                    else:
                        counts[c][w]+= 1
        
        log_prior={} 
        log_likelihood={}  
        for c in classes:
            #total_count = 0
            log_likelihood[c]={}
            num_c = float(len(dic[c]))
            num_tot = float(len(trainingLabel))
            log_prior[c] = math.log((num_c)/(num_tot),2)
            
            class_word_cnt = 0
            for v in vocabulary:
                if v in list(counts[c].keys()):
                    class_word_cnt += counts[c][v]
                #if v in list(counts[c].keys()):
                #class_word_cnt += counts[c][v]

            word_cnt = 0
            for sv in vocabulary:
                if sv in list(counts[c].keys()):
                    word_cnt = counts[c][sv]
                # add-one smoothing 
                log_likelihood[c][sv] = math.log((word_cnt+1)/(class_word_cnt+vocab_size),2)
        self.vocabulary = vocabulary
        self.log_prior=log_prior
        self.log_likelihood = log_likelihood
        #print(log_likelihood)
    

    def estimateLogProbability(self, sentence):
        """
        param sentence: the test sentence, as a single string without label
        return: a dictionary containing log probability for each category
        """

        result={'negative': 0, 'neutral': 0, 'positive':0}
        for c in list(result.keys()):
            result[c]=self.log_prior[c]

            input_words = sentence.split(" ")
            for wi in range(len(input_words)-1):
                wb=input_words[wi]+' '+input_words[wi+1]
                if wb in self.vocabulary:
                   result[c]+=self.log_likelihood[c][wb]
            
        return result
    
    def predictLabel(self, testData):
        predicted_label = []
        for t in testData:
            estimate_result = self.estimateLogProbability(t)
            predicted_label.append(max(estimate_result,key=estimate_result.get))
        return predicted_label

    def testModel(self, predicted_label, real_label):
        """
        param predicted_label: label predicted by the model
        param real_label: label in test dataset
        return: accuracy based on how many predicted labels are consistent with real labels
        """
        tot_cnt = len(predicted_label)
        accurate_cnt = 0
        for p in range(len(predicted_label)):
            # predicted label and real label equal
            if predicted_label[p] == real_label[p]:
               accurate_cnt += 1
         
        accuracy = round(accurate_cnt/tot_cnt, 2)
        
        return accuracy
        

if __name__ == '__main__':
    
    if len(sys.argv) != 5:
        print("Usage: python3 naivebayes.py TOKENIZATION_METHOD(unigram/bigram) TRAIN_DATA TRAIN_LABEL TEST_DATA PREDICTED_LABEL_OUTPUT")
        sys.exit(1)
    
    tokenization_method = sys.arg[1]
    train_text = sys.argv[2]
    label_text = sys.argv[3]
    test_text = sys.argv[4]
    test_label_text = sys.argv[5]
    


    with io.open(train_text, 'r', encoding='utf8') as f:
        train = f.read()
    
    with io.open(label_text, 'r', encoding='utf8') as f:
        label = f.read()

    with io.open(test_text, 'r', encoding='utf8') as f:
        test = f.read()
    
    # preprocess data 
    # cleaning the training data
    # remove URL http:
    data_remove_url = re.sub(r"http\S+","", train)
    # remove @username:
    data_remove_name = re.sub(r"@[^\s]+", "", data_remove_url)
    # remove hashtags and other punct:
    cleaned_data = re.sub(r"([#])|([^ a-zA-Z\n])","", data_remove_name)
              
    all_data = cleaned_data.split("\n")
    all_label = label.split("\n")
    
    # split into training and testing sets
    # 70% training and 30% testing 
    train_data = all_data[:1600]
    train_label= all_label[:1600]
    test_data = all_data[1600:]
    test_label = all_label[1600:]
    
    # preprocess test data
    test_data_remove_url = re.sub(r"http\S+","", test)
    # remove @username:
    test_data_remove_name = re.sub(r"@[^\s]+", "", test_data_remove_url)
    # remove hashtags and other punct:
    cleaned_test_data = re.sub(r"([#])|([^ a-zA-Z\n])","", test_data_remove_name)
    cleaned_test_data = cleaned_test_data.split("\n")
    
 
    model = NaiveBayes(tokenization_method, train_data, train_label)
    get_label = model.predictLabel(test_data)
    evaluation = model.testModel(get_label, test_label)
    prediction = model.predictLabel(cleaned_test_data)
    
    with open(test_label_text, "w") as f:
         f.write("\n".join(prediction))
    




