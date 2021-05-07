!git clone https://github.com/neubig/nlptutorial.git #gitからCloneする

import random
import math
from collections import defaultdict

def word_count(inputs):
  dicts = defaultdict(lambda: 0)
  context_counts = defaultdict(lambda: 0)
  for line in inputs:
    words = line.strip().split(" ")
    words.append("</s>")
    words.insert(0, "<s>")
    for i in range(1, len(words)):
      dicts[words[i-1]+" "+words[i]] += 1
      context_counts[words[i-1]] += 1
      dicts[words[i]] += 1
      context_counts[""] += 1
  for ngram in dicts:
    context = ngram.split(" ")
    context.pop()
    context = ''.join(context)
    dicts[ngram] = dicts[ngram]/context_counts[context]
  return dicts

def calc_entro(dicts, inputs, lambda2):
  lambda1, V, W, H= 0.95, 1000000, 0, 0
  for line in inputs:
    words = line.strip().split(" ")
    words.append("</s>")
    words.insert(0, "<s>")
    for i in range(1, len(words)):
      p1 = lambda1*dicts[words[i]]+(1-lambda1)/V
      p2 = lambda2*dicts[words[i-1]+" "+words[i]]+(1-lambda2)*p1
      H += -math.log(p2, 2)
      W += 1
  return H/W

#必要なファイルを読みこみ
train_input = open('/content/nlptutorial/test/02-train-input.txt')
train_answer = open('/content/nlptutorial/test/02-train-answer.txt')
test_answer = open('/content/nlptutorial/test/01-test-answer.txt')

#プログラムのテスト
t_dict = word_count(train_input)
for foo, bar in sorted(t_dict.items()):
  print("%s %r" % (foo, bar))

for i in range(10, 100, 10):
  with open('/content/nlptutorial/data/wiki-en-test.word') as test_input:
    print("lambda2: {}のとき, entropy = {}".format(i/100, calc_entro(t_dict, test_input, i/100)))

#演習問題
ptrain_input = open('/content/nlptutorial/data/wiki-en-train.word')

pt_dict = word_count(ptrain_input)
for i in range(10, 100, 10):
  with open('/content/nlptutorial/data/wiki-en-test.word') as ptest_input:
    print("lambda2: {}のとき, entropy = {}".format(i/100, calc_entro(pt_dict, ptest_input, i/100)))

#lambda2: 0.1のとき, entropy = 9.968183608330854
#lambda2: 0.2のとき, entropy = 9.814579879571195
#lambda2: 0.3のとき, entropy = 9.75353475358484
#lambda2: 0.4のとき, entropy = 9.748028429401025
#lambda2: 0.5のとき, entropy = 9.788523799888132
#lambda2: 0.6のとき, entropy = 9.877284182445067
#lambda2: 0.7のとき, entropy = 10.028542440615604
#lambda2: 0.8のとき, entropy = 10.281368052356662
#lambda2: 0.9のとき, entropy = 10.767377459346687