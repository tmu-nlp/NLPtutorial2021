'''
viterbi algorithmでは、各エッジは単語、エッジの重みは負の対数確率
(pが大きいほど、―logｐが0に近づく　→最短経路)
グラフの経路は文の分割候補を表し、経路の重みは文のunigramの負対数確率
前向きステップと後ろ向きステップで学習
'''''

import os
import sys
import math
from collections import defaultdict
import subprocess


os.chdir(os.path.dirname(os.path.realpath('__file__')))

def message(text='', CR=False):
    text = '\r' + text if CR else text + '\n'
    sys.stderr.write(text)

def load_uni_probs(path_m):
    probs = defaultdict(float)
    with open(path_m, 'r', encoding='utf-8') as probs_f:
        for line in probs_f:
            word, prob = line.split('\t')
            probs[word] = float(prob)
    return probs


def viterbi(probs, input_path, output_path):
    lambda_1 = 0.95
    lambda_unk = 1 - lambda_1
    V = 1e6

    results = []
    with open(input_path, 'r', encoding='utf-8') as in_f:

        for line in in_f.readlines():
            line = line.strip()
            # p37:forward step
            best_edge = {}
            best_score = {}
            best_edge[0] = None
            best_score[0] = 0
            for word_end in range(1, len(line) + 1):
                best_score[word_end] = float('inf')  # set to inf
                for word_begin in range(word_end):
                    word = line[word_begin : word_end]
                    if word in probs.keys() or len(word) == 1:
                        if word in probs.keys():
                            prob = lambda_1*probs[word] + (lambda_unk) / V
                        # probability of unks
                        else:
                            prob = lambda_unk / V
                        my_score = best_score[word_begin] + (-math.log(prob))
                        if my_score < best_score[word_end]:
                            best_score[word_end] = my_score
                            best_edge[word_end] = (word_begin, word_end)
            # p38:backward step
            words = []
            next_edge = best_edge[len(best_edge) -1]
            while next_edge != None:
                # このエッジの部分文字列を追加
                word = line[next_edge[0] : next_edge[1]]
                words.append(word)
                next_edge = best_edge[next_edge[0]]
            words.reverse()
            results.append(' '.join(words) + '\n')

    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.writelines(results)


if __name__ == '__main__':
    is_test = sys.argv[1:] == ['test']

    if is_test:
        message('[*]test for word segmentation with viterbi_model')
        path_m = '../test/04-model.txt'
        input_path = '../test/04-input.txt'
        output_path = './test_results.word'

    else:
        message('[+]wiki for word segmentation with viterbi_model')
        path_m = './uni_probs.txt'
        input_path = '../data/wiki-ja-test.txt'
        output_path = './wiki_results.word'

    probs = load_uni_probs(path_m)
    viterbi(probs, input_path, output_path)

    if is_test:
        subprocess.run(
            f'diff -s {output_path} ../test/04-answer.txt'.split(),shell=True)

    # 分割精度の評価
    subprocess.run(
        f'perl ../script/gradews.pl\
        ../data/wiki-ja-test.word {output_path}'.split(),shell=True)   # run in cmd, not pycharmTerminal.

    message("[**] Finished!")



'''
Sent Accuracy: 0.00% (/84)            -> have no idea why I got this results 
Word Prec: 68.05% (1866/2742)
Word Rec: 80.88% (1866/2307)
F-meas: 73.92%
Bound Accuracy: 82.73% (2669/3226)

1,2c1,2
< ab c
< b bc
---
> ab c
> b bc
Sent Accuracy: 0.00% (/2)
Word Prec: 25.00% (1/4)
Word Rec: 16.67% (1/6)
F-meas: 20.00%
Bound Accuracy: 33.33% (2/6)'''





