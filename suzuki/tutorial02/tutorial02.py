# -*- coding: utf-8 -*-
#2つのプログラムを作成
#train_bigram: 2-gramモデルを学習
#test_bigram: 2-gramモデルに基づいて評価データのエントロピーを計算

import math
from collections import defaultdict

def train_bigram(target): #2gramと1gramのそれぞれの出現確率を保存
    counts = defaultdict(lambda: 0)
    context_counts = defaultdict(lambda: 0)

    for line in target:
        line = line.strip()
        words = line.split(" ")
        words.append("</s>")
        words.insert(0, "<s>")
        for i in range(1, len(words)):
            counts["{} {}".format(words[i - 1], words[i])] += 1 #2gramの頻度を追加（2gramの分子）
            context_counts[words[i - 1]] += 1 #2gramの分母
            counts[words[i]] += 1 #1gramの分子
            context_counts[""] += 1 #1gramの分母

    f = open('02_model.txt', 'w')

    for ngram, count in counts.items():
        n = ngram.split(" ")
        n.pop(-1)
        s = " ".join(n) #この処理は2gramでは関係ないが,3gram以上では条件つき確率を計算するために必要
        probability = counts[ngram] / context_counts[s]
        f.write("{}\t{:6f}\n".format(ngram, probability))
    
    f.close()
    
    return 0

def test_bigram(model_file, test_file, lambda1, lambda2, V):
    probabilities = defaultdict(lambda: 0)
    H = 0
    W = 0
    
    for line in model_file: #load model
        line = line.strip()
        words = line.split("\t")
        probabilities[words[0]] = float(words[1])
    
    for test_line in test_file:
        test_line = test_line.strip()
        test_words = test_line.split(" ")
        test_words.append("</s>")
        test_words.insert(0, "<s>")
        for i in range(1, len(test_words)):
            p1 = lambda1 * probabilities[test_words[i]] + (1 - lambda1) / V
            p2 = lambda2 * probabilities["{} {}".format(test_words[i - 1], test_words[i])] + (1 - lambda2) * p1
            H += math.log(p2, 2) * (-1)
            W += 1

    print("entropy = {}".format(H / W))

    return 0
        
#テスト
#test_train = open('02-train-input.txt', 'r')
#train_bigram(test_train)
 
#演習
train = open('wiki-en-train.word', 'r')
train_bigram(train)

model = open('02_model.txt', 'r')
test = open('wiki-en-test.word', 'r')
test_bigram(model, test, 0.95, 0.95, 1000000)

model = open('02_model.txt', 'r') #なぜかいちいち宣言しないと W=0 になる
test = open('wiki-en-test.word', 'r')
test_bigram(model, test, 0.95, 0.05, 1000000)

model = open('02_model.txt', 'r')
test = open('wiki-en-test.word', 'r')
test_bigram(model, test, 0.05, 0.95, 1000000)

model = open('02_model.txt', 'r')
test = open('wiki-en-test.word', 'r')
test_bigram(model, test, 0.05, 0.05, 1000000)