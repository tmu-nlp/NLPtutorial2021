import math
from collections import defaultdict
#tutorial01のunigramモデル
from unigram import UnigramModel


def WordSegmentation(model_file, test_file, output_file):

    #unigramの学習
    train_file = '../data/wiki-ja-train.word'
    UnigramModel(train_file, model_file)
    
    #モデル読み込み
    model_dic = defaultdict(lambda: 0)
    with open(model_file) as f:
        for line in f:
            word, prob = line.split()
            model_dic[word] = prob

    #viterbiアルゴリズム
    with open(test_file) as f:
        input = f.readlines()
        
    with open(output_file, 'w') as f:
        for line in input:

            # forward step
            lambda_1 = 0.95; N = 10**6
            best_edge = {}
            best_score = {}
            best_edge[0] = None
            best_score[0] = 0
            for word_end in range(1, len(line) + 1):
                best_score[word_end] = float('inf')
                for word_begin in range(word_end):
                    word = line[word_begin: word_end]
                    if word in model_dic or len(word) == 1:
                        prob = lambda_1 * float(model_dic[word]) + (1 - lambda_1) / N
                        my_score = best_score[word_begin] - math.log(prob, 2)
                        if my_score < best_score[word_end]:
                            best_score[word_end] = my_score
                            best_edge[word_end] = [word_begin, word_end]
                
            #backward step
            words = []
            next_edge = best_edge[len(best_edge) - 1]
            while next_edge != None:
                word = line[next_edge[0]: next_edge[1]]
                words.append(word)
                next_edge = best_edge[next_edge[0]]
            words.reverse()

            f.write(' '.join(words))


if __name__ == '__main__':
    model_file = 'model_txt'
    test_file = '../data/wiki-ja-test.txt'
    output_file = 'my_answer.word'
    WordSegmentation(model_file, test_file, output_file)

"""
Sent Accuracy: 0.00% (/84)
Word Prec: 71.88% (1943/2703)
Word Rec: 84.22% (1943/2307)
F-meas: 77.56%
Bound Accuracy: 86.30% (2784/3226)
"""

#大きなテキストで学習
"""
train_file = '../data/big-ws-model.txt'

if __name__ == '__main__':
    model_file = 'model_txt2'
    test_file = '../data/wiki-ja-test.txt'
    output_file = 'my_answer.word2'
    WordSegmentation(model_file, test_file, output_file)

Sent Accuracy: 0.00% (/84)
Word Prec: 76.65% (1618/2111)
Word Rec: 70.13% (1618/2307)
F-meas: 73.25%
Bound Accuracy: 85.80% (2768/3226)
"""