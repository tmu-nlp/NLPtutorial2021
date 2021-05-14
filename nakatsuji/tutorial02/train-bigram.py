## file ###
#test/02-train-input.txt ---> test/01-train-answer.txt
#data/wiki-en-train.word ---> data/wiki-en-test.wor
writing = "model_file.txt"

import sys
import math
from collections import *

def train_bigram(train_file):
    counts = defaultdict(lambda: 0)
    context_counts = defaultdict(lambda: 0)
    total_counts = 0

    with open(train_file) as train:
        lines = train.readlines()
        for line in lines:
            words = line.strip().split()
            words.append("EOS")
            words.insert(0, "BOS")
            for i in range(1, len(words)):
                counts["".join(words[i-1:i+1])] += 1
                context_counts["\s".join(words[i-1])] += 1
                counts[words[i]] += 1
                context_counts[""] += 1
        counts = dict(sorted(counts.items()))
        return counts, context_counts

if __name__ == "__main__":
    input = sys.argv[1]
    counts, context_counts = train_bigram(input)
    #print(counts)
    with open(writing, "w") as model:
        for ngram, count in counts.items():
            words = ngram.split("/s")
            words.pop()
            context = "".join(words)
            prob = float(counts[ngram]/context_counts[context])
            print("{} {}".format(ngram, prob) , file = model)