#################################
####### Viterbi Algorithm #######
#################################
# author: Yiru Xiong

# The Viterbi algorithm is a dynamic programming algorithm used for finding the most probable sequence of hidden states in a Hidden Markov Model (HMM)

import math
import sys
import time
import numpy as np
from collections import defaultdict

TRANSITION_TAG = "trans"
EMISSION_TAG = "emit"
OOV_WORD = "OOV"         
INIT_STATE = "init"      
FINAL_STATE = "final"    

class Viterbi():
    def __init__(self):
        # transition and emission probabilities

        self.transition = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.emission = defaultdict(lambda: defaultdict(lambda: 1.0))
        # keep track of states to iterate over 
        self.states = set()
        self.POSStates = set()
        # store vocab to check for OOV words
        self.vocab = set()

        # text to run viterbi with
        self.text_file_lines = []
        with open(TEXT_FILE, "r") as f:
            self.text_file_lines = f.readlines()

    def readModel(self):
        # Read HMM transition and emission probabilities
        # Probabilities are converted into LOG SPACE!
        with open(HMM_FILE, "r") as f:
            for line in f:
                line = line.split()

                # Read transition
                # Example line: trans NN NNPS 9.026968067100463e-05
                # Read in states as prev_state -> state
                if line[0] == TRANSITION_TAG:
                    (prev_state, state, trans_prob) = line[1:4]
                    self.transition[prev_state][state] = math.log(float(trans_prob))
                    self.states.add(prev_state)
                    self.states.add(state)

                # Read in states as state -> word
                elif line[0] == EMISSION_TAG:
                    (state, word, emit_prob) = line[1:4]
                    self.emission[state][word] = math.log(float(emit_prob))
                    self.states.add(state)
                    self.vocab.add(word)

        # Keep track of the non-initial and non-final states
        self.POSStates = self.states.copy()
        self.POSStates.remove(INIT_STATE)
        self.POSStates.remove(FINAL_STATE)

    # run Viterbi algorithm and write the output to the output file
    def runViterbi(self):
        result = []
        for line in self.text_file_lines:
            result.append(self.viterbiLine(line))
        #print(result)

        # Print output to file
        with open(OUTPUT_FILE, "w") as f:
            for line in result:
                f.write(line)
                f.write("\n")


    ## Viterbi algorithm 
    # Input: A string representing a sequence of tokens separated by white spaces 
    # Output: A string representing a sequence of POS tags.
    # Note Probability calculations are done in log space. 

    def viterbiLine(self, line):
        words = line.split()

        # TODO: Initialize DP matrix for Viterbi here
        states_list = list(self.states)
        states_size = len(states_list)
        line_len = len(words)
        #      tag1  tag2
        # tags  RBS  RBS
        #       RBS  NN
        #       RBS  TO
        states_comb = np.column_stack((np.repeat(states_list, states_size),np.tile(states_list, states_size)))
        #states_comb = itertools.product(states_list, states_list)
        #prob={0:{INIT_STATE:0.0}}
        prob=defaultdict(lambda: defaultdict(lambda: 1.0))
        prob[0]={INIT_STATE:0.0}
        backpointer=defaultdict(lambda: defaultdict(lambda: 1.0))
        #prob[0]={INIT_STATE:0.0}
        #bestseq=[]
        prevp = 0.0
        alphap = 0.0
        betap = 0.0
        totalp=defaultdict(float)
             
        for (i, word) in enumerate(words):
            i += 1
            # replace unseen words as oov
            if word not in self.vocab:
                word = OOV_WORD

            # TODO: Fill up your DP matrix
            for tag1, tag2 in states_comb:
                if (self.transition[tag2][tag1] != 1.0) and (self.emission[tag1][word] != 1.0):
                    if tag2 in prob[i-1]:
                        alphap=self.transition[tag2][tag1]
                        betap = self.emission[tag1][word]
                        prevp = prob[i-1][tag2]
                        # probability is in log space, so the total p is the sum of three probs
                        totalp = prevp+ alphap + betap
                        if tag1 not in prob[i]:
                            prob[i][tag1] = totalp
                            backpointer[i][tag1] = tag2
                        # prob[i][tag1] exist, compare the prob
                        if totalp > prob[i][tag1]:
                            prob[i][tag1] = totalp
                            backpointer[i][tag1] = tag2

        # find the best final state 
        max_final = 0
        tag = ''
        for s in states_list:
            if (s in prob[line_len]) and (self.transition[s][FINAL_STATE] != 1.0):
                alphap = self.transition[s][FINAL_STATE]
                prevp = prob[line_len][s]
                totalp = prevp + alphap
                if totalp>max_final:
                    max_final = totalp
                    tag = s
             
        # Backtrack and find the optimal sequence.
        if tag != '':
            bestseq=[]
            bestseq.append(tag)
            for t in range(line_len, 1, -1):
                bestseq.append(backpointer[t][tag])
                tag = backpointer[t][tag]
            
            #reverse the best sequence
            bestseq.reverse()
            output=' '.join(bestseq)
        # if cannot proceed, return nothing
        else:    
            output = ''

        return output


# Usage: python3 viterbi.py <HMM_FILE> <TEXT_FILE> <OUTPUT_FILE>

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 viterbi.py HMM_FILE TEXT_FILE OUTPUT_FILE")
        sys.exit(1)
    
    HMM_FILE = sys.argv[1]
    TEXT_FILE = sys.argv[2]
    OUTPUT_FILE = sys.argv[3]
    # Mark start time
    t0 = time.time()
    viterbi = Viterbi()
    viterbi.readModel()
    viterbi.runViterbi()
    # Mark end time
    t1 = time.time()
    print("Time taken to run: {}".format(t1 - t0))

