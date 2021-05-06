from collections import defaultdict
import math
import numpy


class Ngram:
    def __init__(self, lambda_1, lambda_2):
        self.counts = defaultdict(int)                                   #create counts,context_counts
        self.context_counts = defaultdict(int)                           #defaultdict(int): just count
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def trainNgram(self, inputFile):
        model = defaultdict(int)                                         #defaultdict(int): just count
        with open(inputFile, "r") as train:                              
            for line in train:
                line = line.lower().split(" ")                           #'A' to 'a' and split line into words
                line.append("</s>")                                      #add "<s>" at the end of the line
                line.insert(0, "<s>")                                    #insert "<s>" at the beginning of the line

                for i in range(1, len(line)):                            #1~len-1
                    self.counts[" ".join(line[i - 1 : i + 1])] += 1      #Add molecular and denominator of 2-gram
                    self.context_counts[line[i - 1]] += 1
                    self.counts[line[i]] += 1                            #Add molecular and denominator of 1-gram
                    self.context_counts[""] += 1

        for ngram in self.counts:
            context = ngram.split(" ")                                   #split each ngram into words
            context = "".join(context[:-1])                              #"Wi-1 Wi"--{"Wi-1","Wi"}--{"Wi-1"}--"Wi-1"
            probability = self.counts[ngram] / self.context_counts[context]   
            model[ngram] = probability
        return model

    def testNgram(self, modeldic, testFile):
        W = 0
        H = 0
        V = 1000000
        with open(testFile, "r") as test:
            for line in test:
                line = line.lower().split()
                line.append("</s>")
                line.insert(0, "<s>")
                for i in range(1, len(line) - 1):
                    P1 = self.lambda_1 * modeldic[line[i]] + (1 - self.lambda_1) / V      #P1 & P2: 1-gram & 2-gram
                    P2 = (self.lambda_2 * modeldic[" ".join(line[i - 1 : i])]+ (1 - self.lambda_2) * P1)
                    H += math.log(1 / P2, 2)
                    W += 1
        return str(round(H / W, 4))


if __name__ == "__main__":
    trainpath = "/Users/kcnco/github/nlptutorial-master/data/wiki-en-train.word"
    testpath = "/Users/kcnco/github/nlptutorial-master/data/wiki-en-test.word"

    NgramLM = Ngram(0.95, 0.5)                       #range of lambda: 0.05~0.95 (0.05per)
    model = NgramLM.trainNgram(trainpath)
    print("entropy = " + NgramLM.testNgram(model, testpath), end=", ")
    print("when lambda1:0.95, lambda2:0.5")
