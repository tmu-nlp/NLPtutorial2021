### file ###
#test/01-train-input.txt ---> test/01-train-answer.txt
#test/01-test-input.txt  ---> test/01-test-answer.txt

writing = "model_file.word"

import sys
import math
from collections import *

def train_unigram(train_file):
    counts = defaultdict(lambda: 0)
    total_counts = 0

    with open(train_file) as train:
        lines = train.readlines()
        for line in lines:
            words = line.strip().split()
            words.append("EOS")
            for word in words:
                counts[word] += 1
                total_counts += 1
    counts = sorted(counts.items(), key=lambda x:x[0])
    return counts, total_counts

if __name__ == "__main__":
    input = sys.argv[1]
    counts, total_counts = train_unigram(input)
    with open(writing, "w") as model:
        for k, v in counts:
            prob = float(v / total_counts)
            print("{} {}".format(k, prob) , file = model)