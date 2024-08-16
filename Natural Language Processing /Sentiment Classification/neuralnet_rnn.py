######################################################
####### Sentiment Classification based on RNN  #######
######################################################
# author: Yiru Xiong

import torch
from torch import optim, nn
from nltk.tokenize import word_tokenize
import numpy as np
from random import randint
from collections import defaultdict, Counter
import argparse
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('stopwords')

#OUT_HELDOUT_PATH = "/Users/elizabethx/Documents/CMU/Fall 2020/NLP/hw07-handout/HW7/heldout_pred_rnn.txt"
OUT_HELDOUT_PATH = "/you_work_directory/heldout_pred_rnn.txt"

idx2label = ["positive", "neutral", "negative"]
label2idx = {label: idx for idx, label in enumerate(idx2label)}

class ClassifierRunner(object):
    def __init__(self, data, voca_size, rnn_in_dim, rnn_hid_dim):
        self.data = data
        self.clf = Classifier(voca_size, rnn_in_dim, rnn_hid_dim)
        self.optimizer = optim.Adam(self.clf.parameters())
        self.ce_loss = nn.CrossEntropyLoss()

    def run_epoch(self, split):
        """Runs an epoch, during which the classifier is trained or applied
        on the data. Returns the predicted labels of the instances."""

        if split == "dev": self.clf.train()
        else: self.clf.eval()

        labels_pred = []
        for i, (words, label) in enumerate(self.data[split]):
            logit = self.clf(torch.LongTensor(words))

            # Optimize
            if split == "dev":
                loss = self.ce_loss(logit, torch.LongTensor([label]))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            labels_pred.append(idx2label[randint(0, 2)])

        return labels_pred


class Classifier(nn.Module):
    def __init__(self, voca_size, rnn_in_dim, rnn_hid_dim):
        super(Classifier, self).__init__()

        self.rnn_in_dim = rnn_in_dim
        self.rnn_hid_dim = rnn_hid_dim

        # Layers
        self.word2wemb = nn.Embedding(voca_size, rnn_in_dim)
        self.rnn = nn.RNN(rnn_in_dim, rnn_hid_dim)
        self.rnn2logit = nn.Linear(rnn_hid_dim, 3)

    def init_rnn_hid(self):
        """Initial hidden state."""
        return torch.zeros(1, 1, self.rnn_hid_dim)

    def forward(self, words):
        """Feeds the words into the neural network and returns the value
        of the output layer."""
        wembs = self.word2wemb(words) # (seq_len, rnn_in_dim)
        rnn_outs, _ = self.rnn(wembs.unsqueeze(1), self.init_rnn_hid())
                                      # (seq_len, 1, rnn_hid_dim)
        logit = self.rnn2logit(rnn_outs[-1]) # (1 x 3)
        return logit
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-rnn_in_dim", default=100, type=float,
                        help="Dimentionality of RNN inputs")
    parser.add_argument("-rnn_hid_dim", default=30, type=float,
                        help="Dimentionality of RNN hidden state")
    parser.add_argument("-epochs", default=10, type=int,
                        help="Number of epochs")
    args = parser.parse_args()
    
    
    idx2label = ["positive", "neutral", "negative"]
    label2idx = {label: idx for idx, label in enumerate(idx2label)}

    stop = set(stopwords.words('english')) 
    stemmer = nltk.stem.SnowballStemmer('english') 
    print("Reading data...")
    data_raw = defaultdict(list)
    voca_cnt = Counter()
    for text, label in zip(open("/Users/elizabethx/Documents/CMU/Fall 2020/NLP/hw07-handout/HW7/dev_text.txt"), open("/Users/elizabethx/Documents/CMU/Fall 2020/NLP/hw07-handout/HW7/dev_label.txt")):
        # preprocess the data
        # remove URL http:
        text_remove_url = re.sub(r':.*$','', text)
        # remove @username:
        text_remove_name = re.sub(r"@[^\s]+", "", text_remove_url)
        # remove hashtags and other punct:
        cleaned_text = re.sub(r"([#])|([^ a-zA-Z\n])","", text_remove_name)
        words = word_tokenize(cleaned_text.strip())
        cleaned_words = [word for word in words if word.lower() not in stop and word.isalpha()]
        data_raw["dev"].append((cleaned_words, label2idx[label.strip()]))
        voca_cnt.update(cleaned_words)

    for text in open("/Users/elizabethx/Documents/CMU/Fall 2020/NLP/hw07-handout/HW7/heldout_text.txt"):
        # preprocess the data
        # remove URL http:
        test_remove_url = re.sub(r':.*$','', text)
        # remove @username:
        test_remove_name = re.sub(r"@[^\s]+", "", test_remove_url)
        # remove hashtags and other punct:
        cleaned_test = re.sub(r"([#])|([^ a-zA-Z\n])","", test_remove_name)
        test_words = word_tokenize(cleaned_test.strip())
        cleaned_test_words = [word for word in test_words if word.lower() not in stop and word.isalpha()]
        data_raw["heldout"].append((cleaned_test_words, None))

    
    print("Building vocab...")
    word_idx = {"[UNK]": 0}
    for word in voca_cnt.keys():
        word_idx[word] = len(word_idx)
    print("n_voca:", len(word_idx))

    print("Indexing words...")
    data = defaultdict(list)
    for split in ["dev", "heldout"]:
        for words, label in data_raw[split]:
            data[split].append(([word_idx.get(w, 0) for w in words], label))

    print("Running classifier...")
    M = ClassifierRunner(data, len(word_idx), rnn_in_dim=100, rnn_hid_dim=30)
    for epoch in range(10):
        print("Epoch", epoch+1)

        # Train
        M.run_epoch("dev")

        # Test
        with torch.no_grad():
            labels_pred = M.run_epoch("heldout")
        with open(OUT_HELDOUT_PATH, "w") as f:
            f.write("\n".join(labels_pred))


for text, label in zip(open("/content/dev_text.txt"), open("/content/dev_label.txt")):
    # remove URL http:
    text_remove_url = re.sub(r':.*$','', text)
    # remove @username:
    text_remove_name = re.sub(r"@[^\s]+", "", text_remove_url)
    # remove hashtags and other punct:
    cleaned_text = re.sub(r"([#])|([^ a-zA-Z\n])","", text_remove_name)
    words = word_tokenize(cleaned_text.strip())
    cleaned_words = [word for word in words if word.lower() not in stop]
    data_raw["dev"].append((cleaned_words, label2idx[label.strip()]))
    voca_cnt.update(cleaned_words)

