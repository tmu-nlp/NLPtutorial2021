#Google Colabでやっているので少し違うかもしれない
!git clone https://github.com/neubig/nlptutorial.git #gitからCloneする

def word_count(inputs):
  dicts = {}
  for line in inputs:
    line = line.strip()
    words = line.split(" ")
    for word in words:
      if word in dicts.keys():
        dicts[word] += 1
      else:
        dicts[word] = 1
  return dicts

def unit_test(answers, dicts):
  for line in answers:
    line = line.strip()
    words = line.split("	")
    bar = dicts[str(words[0])]
    if int(words[1]) != bar:
      return 0
  return 1

import random
from collections import defaultdict

#必要なファイルを読みこみ
t_input = open('/content/nlptutorial/test/00-input.txt')
t_answer = open('/content/nlptutorial/test/00-answer.txt')

#プログラムのテスト
t_dict = word_count(t_input)

for foo, bar in sorted(t_dict.items()):
  print("%s %r" % (foo, bar))

print(unit_test(t_answer, t_dict))

t_input.close()
t_answer.close()

#演習問題
p_input = open('/content/nlptutorial/data/wiki-en-train.word')
p_dict = word_count(p_input)

#answer = open('/content/wiki-en-answer.txt', 'w')

print("Number of Word: %i" % len(p_dict))
for i in range(10):
  foo, bar = random.choice(list(p_dict.items()))
  print("%s %r" % (foo, bar))



#for foo, bar in sorted(p_dict.items()):
#  answer.writelines("%s %r\n" % (foo, bar))
#  print("%s %r" % (foo, bar))