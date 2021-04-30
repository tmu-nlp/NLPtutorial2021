import sys
import math
from collections import *
import struct

def probabilities(model_file):
    prob = {}
    with open(model_file) as model:
        lines = model.readlines()
        for line in lines:
            line = line.strip().split()
            prob[line[0]] = float(line[1])
    return prob

def test_unigram(input_file, prob):
    lambda1 = 0.95
    lambdaUNK = 1 - lambda1
    V = 10 ** 6
    W = 0
    H = 0
    unk = 0
    with open(input_file) as test_input:
        lines = test_input.readlines()
        for line in lines:
            words = line.strip().split()
            words.append("EOS")
            for word in words:
                W += 1
                P = lambdaUNK / V
                if word in prob:
                    i = float(prob[word])
                    P += lambda1 * i
                else:
                    unk += 1
                    H += -math.log(P, 2)
    entropy = H / W
    coverage = (W - unk) / W

    return entropy, coverage

if __name__ == "__main__":
    model_file = sys.argv[1]
    test_file = sys.argv[2]
    prob = probabilities(model_file)
    entropy, coverage = test_unigram(test_file, prob)
    print("entropy : {}".format(entropy))
    print("coverage: {}".format(coverage))