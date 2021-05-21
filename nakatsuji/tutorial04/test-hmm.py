import math, sys
from collections import *

def load_model(model):
    transition = {}
    emission = defaultdict(lambda:0)
    possible_tags = {}

    with open(model, encoding='utf-8') as f:
        for line in f.readlines():
            line.strip()
            type, context, word, prob = line.split(' ')
            possible_tags[context] = 1
            if type == 'T':
                transition[f'{context} {word}'] = float(prob)
            else:
                emission[f'{context} {word}'] = float(prob)

    return transition, emission, possible_tags


    return 0
def test_hmm(transition, emission, possible_tags, input, write_name):
    Lamda = 0.95
    V = 10 ** 6
    with open(input) as f, open(write_name, 'w') as out:
        for line in f.readlines():
            line = line.strip()

            # 前向き
            words = line.split(' ')
            l = len(words)
            best_score = {}
            best_edge = {}
            best_score['0 <s>'] = 0 # <s>から始まる
            best_edge['0 <s>'] = None

            for i in range(l):
                for prev in possible_tags.keys():
                    for next in possible_tags.keys():
                        if f'{i} {prev}' in best_score and f'{prev} {next}' in transition:
                            score = best_score[f'{i} {prev}'] - math.log(transition[f'{prev} {next}']) - math.log(Lamda * emission[f'{next} {words[i]}'] + (1 - Lamda)/V)
                            if f'{i+1} {next}' not in best_score or best_score[f'{i+1} {next}'] > score:
                                best_score[f'{i+1} {next}'] = score
                                best_edge[f'{i+1} {next}'] = f'{i} {prev}'
            
            for tag in possible_tags.keys():
                if  f'{l} {tag}' in best_score and f'{tag} </s>' in transition:
                    score = best_score[f'{l} {tag}'] - math.log2(transition[f'{tag} </s>'])
                    if f'{l+1} </s>' not in best_score or best_score[f'{l+1} </s>'] > score:
                        best_score[f'{l+1} </s>'] = score
                        best_edge[f'{l+1} </s>'] = f'{l} {tag}'
            
            #後ろ向き
            tags = []
            next_edge = best_edge[f"{l+1} </s>"]
            while next_edge != '0 <s>':
                position, tag = next_edge.split()
                tags.append(tag)
                next_edge = best_edge[next_edge]
            
            #
            print(" ".join(tags[::-1]), file=out)




if __name__ == "__main__":
    path = '/Users/michitaka/lab/NLP_tutorial/nlptutorial/'
    #テスト
    #model = 'model.txt'
    #transition, emission, possible_tags = load_model(model)
    #input = path + 'test/05-test-input.txt'
    #answer = path + 'test/05-test-answer.txt'
    #test_hmm(transition, emission, possible_tags, input, 'my_answer')
    #演習
    model = 'wiki_model.txt'
    transition, emission, possible_tags = load_model(model)
    input = path + 'data/wiki-en-test.norm'
    #answer = path + 'test/05-test-answer.txt'
    test_hmm(transition, emission, possible_tags, input, 'my_answer.pos')


'''
Accuracy: 90.82% (4144/4563)

Most common mistakes:
NNS --> NN      45
NN --> JJ       27
NNP --> NN      22
JJ --> DT       22
VBN --> NN      12
JJ --> NN       12
NN --> IN       11
NN --> DT       10
NNP --> JJ      8
VBP --> VB      7
'''