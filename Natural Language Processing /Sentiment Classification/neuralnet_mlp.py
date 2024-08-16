
######################################################
####### Sentiment Classification based on MLP ########
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
nltk.download('stopwords')


OUT_HELDOUT_PATH = "/your_work_directory/heldout_pred_rnn.txt"


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
            max_idx = np.argmax(logit.detach().numpy())

            # Optimize
            if split == "dev":
                loss = self.ce_loss(logit, torch.LongTensor([label]))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            labels_pred.append(idx2label[max_idx])

        return labels_pred


class Classifier(nn.Module):
    def __init__(self, voca_size, rnn_in_dim, rnn_hid_dim):
        super(Classifier, self).__init__()

        self.rnn_in_dim = rnn_in_dim
        self.rnn_hid_dim = rnn_hid_dim

        # Layers
        self.fc1 = nn.Linear(rnn_in_dim, rnn_hid_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(rnn_hid_dim, 3)
    
        #self.rnn2logit = nn.Linear(rnn_hid_dim, 3)

    def init_rnn_hid(self):
        """Initial hidden state."""
        return torch.zeros(1, 1, self.rnn_hid_dim)

    def forward(self, words):
        """Feeds the words into the neural network and returns the value
        of the output layer."""
        fc1_out = self.fc1(words)
        relu_out = self.relu(fc1_out)
        fc2_out = self.fc2(relu_out)
        logit = fc2_out
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
    for text, label in zip(open("dev_text.txt"), open("dev_label.txt")):
        # preprocess the data
        # remove URL http:
        text_remove_url = re.sub(r':.*$','', text)
        # remove @username:
        text_remove_name = re.sub(r"@[^\s]+", "", text_remove_url)
        # remove hashtags and other punct:
        cleaned_text = re.sub(r"([#])|([^ a-zA-Z\n])","", text_remove_name)
        words = word_tokenize(cleaned_text.strip())
        #cleaned_words = [word for word in words if word.lower() not in stop]
        data_raw["dev"].append((words, label2idx[label.strip()]))
        voca_cnt.update(words)

    for text in open("heldout_text.txt"):
        # preprocess the data
        # remove URL http:
        test_remove_url = re.sub(r':.*$','', text)
        # remove @username:
        test_remove_name = re.sub(r"@[^\s]+", "", test_remove_url)
        # remove hashtags and other punct:
        #cleaned_test = re.sub(r"([#])|([^ a-zA-Z\n])","", test_remove_name)
        cleaned_test = test_remove_name
        test_words = word_tokenize(cleaned_test.strip())
        #cleaned_test_words = [word for word in test_words if word.lower() not in stop]
        data_raw["heldout"].append((test_words, None))


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
    M = ClassifierRunner(data, len(word_idx), rnn_in_dim = 100, rnn_hid_dim = 30)
    for epoch in range(10):
        print("Epoch", epoch+1)

        # Train
        M.run_epoch("dev")

        # Test
        with torch.no_grad():
            labels_pred = M.run_epoch("heldout")
        with open(OUT_HELDOUT_PATH, "w") as f:
            f.write("\n".join(labels_pred))
