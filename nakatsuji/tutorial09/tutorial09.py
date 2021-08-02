from tqdm import tqdm
import random
import math
from collections import defaultdict


NUM_TOPICS = 3
FILEPATH = "../files/data/wiki-en-documents.word"

ALPHA = 0.01
BETA = 0.01

def sampleone(probs):
    z = sum(probs)
    remaining = uniform(0, z)
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
    exit()

def add_counts(word, topic, doc_id, amounts, xcnts, ycnts):
    xcnts[f"{topic}"] += 1
    xcnts[f"{word}|{topic}"] += 1

    ycnts[f"{doc_id}"] += 1
    ycnts[f"{topic}|{doc_id}"] += 1


def initialize(train_file, num_topics):
    with open(train_file, 'r', encoding='utf-8') as tr_file:
        xcorpus = []
        ycorpus = []
        xcounts = defaultdict(int)
        ycounts = defaultdict(int)
        num_wordtype = set()
        for line in tr_file:
            doc_id = len(xcorpus)
            words = line.strip().split()
            topics = []
            for word in words:
                topic = random.randint(0, num_topics)
                topics.append(topic)
                xcounts, ycounts = add_counts(
                    xcounts, ycounts, word, topic, doc_id, 1)
                num_wordtype.add(word)
            xcorpus.append(words)
            ycorpus.append(topics)
    return xcorpus, ycorpus, xcounts, ycounts, len(num_wordtype)



def main():
    xcorps, ycorps, xcnts, ycnts, num_words = initialize()
    ll = 0

    for i in tqdm(range(len(xcorps)), desc="sent"):
        for j in range(len(xcorps[i])):
            x, y = xcorps[i][j], ycorps[i][j]
            add_counts(x, y, i, -1, xcnts, ycnts)

            probs = []
            for k in range(NUM_TOPICS):
                x_prob = (xcnts[f"{x}|{k}"] + ALPHA) / (xcnts[k] + ALPHA * num_words)
                y_prob = (ycnts[f"{k}|{i}"] + BETA) / (ycnts[i] + BETA * NUM_TOPICS)
                probs.append(x_prob * y_prob)
        new_y = sample_one(probs)
        ll += log(probs[new_y])
        add_counts(x, new_y, i, 1, xcnts, ycnts)
        ycorps[i][j] = new_y

    print(ll)

if __name__ == "__main__":
    main()