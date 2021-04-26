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

def calc_entro(dicts, inputs):
  lambd, V, W, H, unk = 0.95, 1000000, 0, 0, 0
  for line in inputs:
    words = line.strip().split(" ")
    words.append("</s>")
    for word in words:
      W += 1
      P = (1-lambd)/V
      if word in dicts.keys():
        P += lambd * dicts[word]
      else:
        unk += 1
      H += -math.log(P, 2)
  return H/W, (W-unk)/W

#必要なファイルを読みこみ
train_input = open('/content/nlptutorial/test/01-train-input.txt')
train_answer = open('/content/nlptutorial/test/01-train-answer.txt')
test_input = open('/content/nlptutorial/test/01-test-input.txt')
test_answer = open('/content/nlptutorial/test/01-test-answer.txt')

#プログラムのテスト
t_dict = word_count(train_input)
for foo, bar in sorted(t_dict.items()):
  print("%s %r" % (foo, bar))

print("entropy = %f\ncoverage = %f" % calc_entro(t_dict, test_input))

#演習問題
ptrain_input = open('/content/nlptutorial/data/wiki-en-train.word')
ptest_input = open('/content/nlptutorial/data/wiki-en-test.word')
pt_dict = word_count(ptrain_input)
print("entropy = %f\ncoverage = %f" % calc_entro(pt_dict, ptest_input))