from collections import defaultdict
import random
import math
import sys
import string


class Tutorial09:
    xcorpus = []
    ycorpus = []
    xcounts = defaultdict(int)
    ycounts = defaultdict(int)
    words = defaultdict(int)

    alpha = 0.01
    beta  = 0.01

    def __init__(self, lines, num_topics = 20):
        self.num_topics = num_topics
        ptable = str.maketrans(string.punctuation, ' '*len(string.punctuation))

        for line in lines:
            pline = line.translate(ptable)
            docid = len(self.xcorpus)
            topics = []
            words = pline.split()
            for word in words:
                if word == 'bakufumade':
                    print('detect')
                    print(line)
                self.words[word] += 1
                topic = random.randrange(self.num_topics)
                topics.append(topic)
                self.add_counts(word, topic, docid, 1)
            self.xcorpus.append(words)
            self.ycorpus.append(topics)

    def add_counts(self, word, topic, docid, amount):
        self.xcounts[f'{topic}'] += amount
        self.xcounts[f'{word}|{topic}'] += amount
        self.ycounts[f'{docid}'] += amount
        self.ycounts[f'{topic}|{docid}'] += amount

        if (self.xcounts[f'{topic}'] < 0):
            raise Exception('error1')

    def sample_one(self, probs):
        z = sum(probs)
        remaining = random.uniform(0, z)
        for i in range(len(probs)):
            remaining -= probs[i]
            if remaining <= 0:
                return i
        raise Exception('error2')

    def sampling_inner(self):
        ll = 0
        for i in range(len(self.xcorpus)):
            for j in range(len(self.xcorpus[i])):
                x = self.xcorpus[i][j]
                y = self.ycorpus[i][j]
                self.add_counts(x, y, i, -1)
                probs = []
                for k in range(self.num_topics):
                    p_xk = (self.xcounts[f'{x}|{k}'] + self.alpha) / \
                        (self.xcounts[k] + self.alpha * len(self.words))
                    p_ky = (self.ycounts[f'{k}|{y}'] + self.beta) / \
                        (self.ycounts[y] + self.beta * self.num_topics)
                    probs.append(p_xk * p_ky)
                new_y = self.sample_one(probs)
                ll += math.log(probs[new_y])
                self.add_counts(x, new_y, i, 1)
                self.ycorpus[i][j] = new_y
        return ll

    def sampling(self, iters):
        for i in range(iters):
            self.sampling_inner()

    def show_result(self):
        result = defaultdict(lambda: defaultdict(int))
        for i in range(len(self.xcorpus)):
            for j in range(len(self.xcorpus[i])):
                result[self.ycorpus[i][j]][self.xcorpus[i][j]] += 1
        for i in range(self.num_topics):
            print(f'Topic{i}:')
            strings = ''
            for word, num in sorted(result[i].items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f'    {word} ({num})')


if __name__ == '__main__':
    with open('data/wiki-en-documents.word', 'r') as f:
        lines = f.readlines()
    x = Tutorial09(lines, 20)
    x.sampling(10)
    x.show_result()
