##################################
####### TRAIN A HMM  MODEL #######
##################################
# author: Yiru Xiong

# HMMs are probabilistic models used to describe systems that follow a Markov process with hidden states

import sys
import re

from collections import defaultdict

class HMMTrain():
    def __init__(self, TAG_FILE, TOKEN_FILE, OUTPUT_FILE):
        self.TAG_FILE = TAG_FILE
        self.TOKEN_FILE = TOKEN_FILE 
        self.OUTPUT_FILE = OUTPUT_FILE
        #Vocabulary
        self.vocab = {}
        self.OOV_WORD = "OOV"
        self.INIT_STATE = "init"
        self.FINAL_STATE = "final"
        #Transition and emission probabilities
        self.emissions = {}
        self.transitions = {}
        self.transitions_total = defaultdict(lambda: 0)
        self.emissions_total = defaultdict(lambda: 0)


    # train the model 
    # The training data should consist of one line per sequence, with states or symbols separated by whitespace and no trailing whitespace.
    def train(self):
        # Read from tag file and token file. 
        with open(self.TAG_FILE) as tag_file, open(self.TOKEN_FILE) as token_file:
            for tag_string, token_string in zip(tag_file, token_file):
                tags = re.split("\s+", tag_string.rstrip())
                tokens = re.split("\s+", token_string.rstrip())
                pairs = zip(tags, tokens)

                # Starts off with initial state
                prevtag = self.INIT_STATE

                for (tag, token) in pairs:

                    # Note: to handle out-of-vocabulary (OOV) - the first time we see *any* word token, we pretend it is an OOV.  
                    # this lets our model decide the rate at which new words of each POS-type should be expected 
                    # (e.g., high for nouns,low for determiners).

                    if token not in self.vocab:
                        self.vocab[token] = 1
                        token = self.OOV_WORD

                    # keep track of transition and emission probabilities 
                    if tag not in self.emissions:
                        self.emissions[tag] = defaultdict(lambda:0)
                    self.emissions[tag][token] += 1
                    
                    if prevtag not in self.transitions:
                        self.transitions[prevtag] = defaultdict(lambda:0)
                    self.transitions[prevtag][tag] += 1
                    
                    self.emissions_total[tag] += 1
                    self.transitions_total[prevtag] += 1
                    
                    prevtag = tag

                if prevtag not in self.transitions:
                    self.transitions[prevtag] = defaultdict(lambda: 0)

                self.transitions[prevtag][self.FINAL_STATE] += 1
                self.transitions_total[prevtag] += 1

    # calculate the transition probability prevtag -> tag
    def calculate_transition_prob(self, prevtag, tag):
        alpha = float(self.transitions[prevtag][tag]/self.transitions_total[prevtag])      
        return alpha

    #calculate the probability of emitting token given tag
    def calculate_emission_prob(self, tag, token):
            #for token in self.emissions[tag]:
        beta = float(self.emissions[tag][token]/self.emissions_total[tag])
        return beta

    # Write the model to an output file.
    def writeResult(self):
        with open(self.OUTPUT_FILE, "w") as f:
            for prevtag in self.transitions:
                for tag in self.transitions[prevtag]:
                    f.write("trans {} {} {}\n"
                        .format(prevtag, tag, self.calculate_transition_prob(prevtag, tag)))

            for tag in self.emissions:
                for token in self.emissions[tag]:
                    f.write("emit {} {} {}\n"
                        .format(tag, token, self.calculate_emission_prob(tag, token)))


# Usage: python3 train_hmm.py <HMM_FILE> <TEXT_FILE> <OUTPUT_FILE>
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 train_hmm.py TAG_FILE TOKEN_FILE OUTPUT_FILE")
        sys.exit(1)

    # Input Files
    TAG_FILE = sys.argv[1]
    TOKEN_FILE = sys.argv[2]
    OUTPUT_FILE = sys.argv[3]
    

    model = HMMTrain(TAG_FILE, TOKEN_FILE, OUTPUT_FILE)
    model.train()
    model.writeResult()



