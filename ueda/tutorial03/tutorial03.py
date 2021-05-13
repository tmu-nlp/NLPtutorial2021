!git clone https://github.com/neubig/nlptutorial.git #gitからCloneする

import random
import math
from collections import defaultdict

def word_count(inputs):
  dicts = defaultdict(lambda: 0)
  total_words = 0
  for line in inputs:
    words = line.strip().split(" ")
    words.append("</s>")
    for word in words:
      dicts[word] += 1
      total_words += 1
  dicts = {foo: bar / total_words for foo, bar in dicts.items()}
  return dicts

def Viterbi(dicts, inputs):
  answer = open("/content/my_answer.word", "w", encoding="utf-8")
  lambd, V = 0.95, 1000000
  for line in inputs:
    best_edge = {}
    best_score = {}
    best_edge[0] = None
    best_score[0] = 0
    for word_end in range(1, len(line)):
      best_score[word_end] = 10e+10
      for word_begin in range(0, len(line)-1):
        word = line[word_begin:word_end]
        if word in dicts or len(word) == 1:
          prob= (1-lambd)/V
          if word in dicts:
            prob += lambd*float(dicts[word])
          my_score = best_score[word_begin] - math.log(prob)
          if my_score < best_score[word_end]:
            best_score[word_end] = my_score
            best_edge[word_end] = (word_begin, word_end)
    words = []
    next_edge = best_edge[len(best_edge)-1]
    while next_edge != None:
      word = line[next_edge[0]:next_edge[1]]
      words.append(word)
      next_edge = best_edge[next_edge[0]]
    words.reverse()
    answer.write(' '.join(words)+'\n')
    print(' '.join(words))

#必要なファイルを読みこみ
train_model = open('/content/nlptutorial/test/04-model.txt', encoding="utf-8")
test_input = open('/content/nlptutorial/test/04-input.txt', encoding="utf-8")
test_answer = open('/content/nlptutorial/test/04-answer.txt', encoding="utf-8")

#プログラムのテスト
t_dicts = defaultdict(lambda: 0)
for line in train_model:
  words = line.strip().split("\t")
  t_dicts[words[0]] = words[1]
Viterbi(t_dicts, test_input)

#演習問題
pt_model = open('/content/nlptutorial/data/wiki-ja-train.word')
big_model = open('/content/nlptutorial/data/big-ws-model.txt')
pt_input = open('/content/nlptutorial/data/wiki-ja-test.txt')
pt_dict = word_count(pt_model)
big_dicts = defaultdict(lambda: 0)
for line in big_model:
  words = line.strip().split("\t")
  big_dicts[words[0]] = words[1]
Viterbi(big_dicts, pt_input)

"""Sent Accuracy: 23.81% (20/84)
Word Prec: 71.88% (1943/2703)
Word Rec: 84.22% (1943/2307)
F-meas: 77.56%
Bound Accuracy: 86.30% (2784/3226)

Sent Accuracy: 17.86% (15/84)
Word Prec: 89.63% (2066/2305)
Word Rec: 89.55% (2066/2307)
F-meas: 89.59%
Bound Accuracy: 94.54% (3050/3226)
"""