# -*- coding: utf-8 -*-
#2つのプログラムを作成
#train-unigram: 1-gram モデルを学習
#test-unigram: 1-gram モデルを読み込み、エントロピーとカバレージを計算

import sys
import math
from collections import defaultdict

def train_unigram(target):
    counts = defaultdict(lambda: 0) #単語と出現個数を記録する
    total_count = 0 #総単語数（被りあり）

    for line in target:
        line = line.strip()
        words = line.split(" ")
        words.append("</s>")

        for word in words: #辞書の作成と総単語数の数え上げ
            counts[word] += 1
            total_count += 1
    
    list_counts = sorted(counts.items())

    f = open('01_model.txt', 'w')

    for word, count in list_counts:
        probability = float(count) / total_count #確率を計算
        f.write('{} {:6f}\n'.format(word, probability)) #小数点6桁以下は省略して記録
    
    return 0

def test_unigram(model_file, test_file, lambda_unk, V): #未知語確率を設定
    probabilities = {}
    word_count = 0 #test file 内の単語数
    H = 0 #対数尤度
    unk = 0 #未知語の数

    for line in model_file: #モデル読み込み
        line = line.strip()
        l = line.split(" ")
        probabilities[l[0]] = float(l[1].replace('.', '')) / 1000000

    for line in test_file:
        line = line.strip()
        words = line.split(" ")
        words.append("</s>")

        for word in words:
            word_count += 1
            p = lambda_unk / V #未知語確率のみで初期化
            
            if word in probabilities: #未知語でない場合確率を追加
                p += (1 - lambda_unk) * probabilities[word]
            else:
                unk += 1
            H += math.log(p, 2) * (-1)
    
    entropy = H / word_count
    coverage = float(word_count - unk) / word_count

    return entropy, coverage


#テスト
test_train = open("./01-train-input.txt","r")
train_unigram(test_train)

test_test = open("./01-test-input.txt","r")
test_model = open("./01_model.txt")
e, c = test_unigram(test_model, test_test, 0.05, 1000000)

print("entropy  = {}".format(e)) #entropy = 6.709899
print("coverage = {}".format(c)) #coverage = 0.800000

#演習
train = open("./wiki-en-train.word","r")
train_unigram(train)

test = open("./wiki-en-test.word", "r")
model = open("./01_model.txt")
e, c = test_unigram(model, test, 0.05, 1000000)

print("entropy  = {}".format(e))
print("coverage = {}".format(c))