from collections import defaultdict
import math


def beam_search(words, possible_tags, transition, emission, beam_size):
    LAMBDA = 0.95
    V = 10**6
    words.append('</s>')
    best_score = defaultdict(lambda: 10**10)
    best_score[f'0 <s>'] = 0
    best_edge = {'0 <s>': None}
    active_tags = {0: ['<s>']}

    for i in range(len(words)):
        my_best = {}
        for prev in active_tags[i]:
            for nxt in possible_tags:
                i_prev = f'{i} {prev}'
                prev_nxt = f'{prev} {nxt}'
                word_nxt = f'{words[i]} {nxt}'
                if i_prev in best_score.keys() and prev_nxt in transition:
                    score = best_score[i_prev] \
                            - math.log(transition[prev_nxt]) \
                            - math.log(LAMBDA*emission[word_nxt] + (1-LAMBDA)/V)
                    i_1_nxt = f'{i+1} {nxt}'
                    if best_score[i_1_nxt] > score:
                        best_score[i_1_nxt] = score
                        best_edge[i_1_nxt] = i_prev
                        my_best[nxt] = score
        
        active_tags[i+1] = []
        for tag in sorted(my_best.items(), key=lambda x: x[1]):
            active_tags[i+1].append(tag[0])
            if len(active_tags[i+1]) == beam_size:
                break

    pred_tags = []
    next_edge = best_edge[f'{len(words)} </s>']
    while next_edge != '0 <s>':
        tag = next_edge.split(' ')[1]
        pred_tags.append(tag)
        next_edge = best_edge[next_edge]
    pred_tags.reverse()
    return pred_tags


def train_hmm(trainfile):
    emit = defaultdict(lambda: 0)
    trans = defaultdict(lambda: 0)
    context = defaultdict(lambda: 0)
    possible_tags = set()
    possible_tags.add('</s>')

    for line in open(trainfile):
        word_tags = line.strip('\n').split(' ')
        prev_tag = '<s>'
        context[prev_tag] += 1
        for word_tag in word_tags:
            word, tag = word_tag.split('_')
            word = word.lower()

            trans[f'{prev_tag} {tag}'] += 1
            emit[f'{word} {tag}'] += 1
            context[tag] += 1
            possible_tags.add(tag)

            prev_tag = tag

        trans[f'{prev_tag} </s>'] += 1
        emit[f'</s> </s>'] += 1
        context['</s>'] += 1

    transition = defaultdict(float)
    for key, trans_cnt in trans.items():
        prev = key.split(' ')[0]
        transition[key] = trans_cnt / context[prev]

    emission = defaultdict(float)
    for key, emit_cnt in emit.items():
        prev = key.split(' ')[1]
        emission[key] = emit_cnt / context[prev]

    return emission, transition, possible_tags


def test_hmm(testfile, emission, transition, possible_tags):
    with open('my_answer.pos', 'w') as out_f:
        for line in open(testfile):
            words = line.strip().split()
            pred_tags = beam_search(words, possible_tags, transition, emission, beam_size=10)
            out_f.write(' '.join(pred_tags)+'\n')

    
if __name__ == '__main__':
    trainfile = '../data/wiki-en-train.norm_pos'
    testfile = '../data/wiki-en-test.norm'
    emission, transition, possible_tags = train_hmm(trainfile)
    test_hmm(testfile, emission, transition, possible_tags)

'''
#beam size = 3
Accuracy: 84.94% (3876/4563)
Most common mistakes:
NNS --> NN      53
NNP --> NN      38
NN --> JJ       37
-RRB- --> NN    35
JJ --> DT       32
NNP --> DT      27
RB --> JJ       26
NN --> DT       25
-LRB- --> IN    19
IN --> DT       18


#beam size = 5
Accuracy: 85.14% (3885/4563)
Most common mistakes:
NNS --> NN      47
NNP --> NN      41
-RRB- --> NN    33
JJ --> DT       32
NN --> JJ       31
RB --> NN       26
NN --> DT       24
-LRB- --> IN    21
NNP --> DT      20
IN --> DT       18


#beam size = 10
Accuracy: 85.51% (3902/4563)
Most common mistakes:
NNS --> NN      45
NNP --> NN      36
JJ --> DT       31
NN --> JJ       31
-RRB- --> NN    31
RB --> NN       27
NN --> DT       24
NNP --> DT      21
-LRB- --> IN    20
IN --> DT       18
'''