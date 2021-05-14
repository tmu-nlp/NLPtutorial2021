import sys
import math
from collections import *
import struct

#def probabilities(model_file):
#    prob = {}
#    with open(model_file) as model:
#        lines = model.readlines()
#        for line in lines:
#            line = line.strip().split()
#            prob[line[0]] = float(line[1])
#    return prob
prob = defaultdict(lambda: 0)
def Prob(model_file):
    with open(model_file) as model_file:
        lines = model_file.readlines()
        for line in lines:
            line = line.strip().split()
            prob[line[0]] = float(line[1])
    return 0

def test_bigram(input_file):
    #lambda1 = 0.95
    #lambdaUNK = 1 - lambda1
    #V = 10 ** 6
    #W = 0
    #H = 0
    #unk = 0
    min_entropy = float("inf")
    with open(input_file) as test_input:
        lines = test_input.readlines()
        for lambda1 in range(5, 100, 5):
            for lambda2 in range(5, 100,5):
                lambda1 = lambda1 / 100
                lambda2 = lambda2 / 100
                V, W, H = 10**6, 0, 0
                
                for line in lines:
                    words = line.strip().split()
                    words.append("EOS")
                    words.insert(0, "BOS")
                    for i in range(1, len(words)):
                        P1 = lambda1 * prob[words[i]] + (1 - lambda1) / V
                        P2 = lambda2 * prob["\s".join(wods[i-1:i+1])] + (1 - lambda2) * P1
                        H += -math.log(P2, 2)
                        W += 1
            entropy = H / W
            if entropy < min_entropy:
                min_entropy = entropy
                min_lambda1, min_lambda2 = lambda1, lambda2

    return min_entropy, min_lambda1, min_lambda2

if __name__ == "__main__":
    model_file = sys.argv[1]
    #test_file = sys.argv[2]
    #prob = probabilities(model_file)
    Prob(model_file)
    entropy, lambda1, lambda2 = test_bigram(model_file)
    print("entropy : {}".format(entropy))
    print("lambda1: {}".format(lambda1))
    print("lambda1: {}".format(lambda2))