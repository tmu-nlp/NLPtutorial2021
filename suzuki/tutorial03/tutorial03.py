# -*- coding: utf-8 -*-
#単語分割プログラムを作成


import math
from collections import defaultdict
from tutorial01_03 import train_unigram

#tutorial01_03.pyを用いて1gramと確率を記録した03_model.txtを作成

def word_segmentation(model, target, lambda_unk, V): #model:モデルファイル target:テストファイル lambda,N:未知語確率
    dict_p = defaultdict(lambda: 0) #単語とその確率の辞書
    f = open('my_answer.txt', 'w')

    for line in model: #load a map of unigram probabilities
        line = line.strip()
        l = line.split("\t")
        dict_p[l[0]] = l[1]
    
    for line in target:
        line = line.strip()
        #python3ではリテラル文字列がデフォルトでunicodeらしい。unicode()は使えなかった。str()ではデコードが不要なのでエラーが出る
        best_edge = [None]
        best_score = [0]

        #前向きステップ
        for word_end in range(1, len(line) + 1, 1): #for each node in the graph
            best_score.append(10 ** 10) #best_score[word_end] = 10 * 10 大きな値に設定

            for word_begin in range(0, word_end, 1):
                word = line[word_begin : word_end] #部分文字列を取得

                if word in dict_p or len(word) == 1: #既知語か長さ1
                    prob = lambda_unk / V

                    if word in dict_p:
                        prob += (1 - lambda_unk) * float(dict_p[word])
                    
                    my_score = best_score[word_begin] - math.log(prob, 2)

                    if my_score < best_score[word_end]:
                        best_score[word_end] = my_score
                        best_edge.append((word_begin, word_end))
        
        #後ろ向きステップ
        words = []
        next_edge = best_edge[len(best_edge) - 1]

        while next_edge != None:
            word = line[next_edge[0] : next_edge[1]] #行中の部分文字列
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        
        words.reverse()

        s = ' '.join(words)
        f.write("{}\n".format(s))
    
    f.close()
    
    return 0

train = open('wiki-ja-train.word', 'r')
train_unigram(train)
model = open('03-model.txt', 'r')
test = open('wiki-ja-test.txt', 'r')
word_segmentation(model, test, 0.05, 1000000)

#結果
#Sent Accuracy: 21.43% (18/84)
#Word Prec: 66.02% (1679/2543)
#Word Rec: 77.41% (1679/2169)
#F-meas: 71.26%
#Bound Accuracy: 81.38% (2465/3029)