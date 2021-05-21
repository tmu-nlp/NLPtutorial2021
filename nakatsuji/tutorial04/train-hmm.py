import math, sys
from collections import *

def train_hmm(input, answer, model_name):
    transition = defaultdict(lambda: 0)
    emission = defaultdict(lambda: 0)
    possoble_tags = defaultdict(lambda: 0)

    with open(input, encoding='utf-8') as f:
        for line in f.readlines():
            previous = '<s>'
            w_tags = line.rstrip().split(' ')
            possoble_tags[previous] += 1
            for w_tag in w_tags:
                word, tag = w_tag.split('_')
                transition[previous + ' ' + tag] += 1
                possoble_tags[tag] += 1
                emission[tag + ' ' + word] += 1
                previous = tag
                transition[previous + ' </s>'] += 1
    with open(model_name + '.txt', 'w') as f:
        for k, v in sorted(transition.items()):
            pre, word = k.split(' ')
            writing = 'T ' + k  + ' ' + str(v / possoble_tags[pre])
            f.write(writing+'\n')
            print(writing)

        for k, v in sorted(emission.items()):
            tag, word = k.split(' ')
            writing = 'E ' + k  + ' ' + str(v / possoble_tags[tag])
            f.write(writing+'\n')
            print(writing)

if __name__ == "__main__":
    path = '/Users/michitaka/lab/NLP_tutorial/nlptutorial/'
    #テスト
    #input = path + 'test/05-train-input.txt'
    #answer = path + 'test/05-train-answer.txt'
    #train_hmm(input, answer, 'model')
    #演習
    input = path + 'data/wiki-en-train.norm_pos'
    answer = path + 'test/05-train-answer.txt'
    train_hmm(input, answer, 'wiki_model')