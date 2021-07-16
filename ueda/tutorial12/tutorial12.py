# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1N5VTrFTy3-pVgvZEDm_rR8jTeonkitJ_
"""

!git clone https://github.com/neubig/nlptutorial.git

from collections import defaultdict
from tqdm import tqdm_notebook as tqdm

class hmm_percep():
  def __init__(self):
    self.w = defaultdict(int)
    self.transition = defaultdict(int)
    self.possible_tags = {'<s>', '</s>'}

  def create_features(self, X, Y):
    phi = defaultdict(int)
    for i in range(0, len(Y)+1):
      if i == 0: first_tag = "<s>"
      else: first_tag = Y[i-1]
      if i == len(Y): next_tag = "</s>"
      else: next_tag = Y[i]
      for key in self.create_trans(first_tag, next_tag):
        phi[key] += 1
    for i in range(0, len(Y)):
      for key in self.create_emit(Y[i], X[i]):
        phi[key] += 1
    return phi
  
  def update_weight(self, dic, sign):
    for key, value in dic.items():
      self.w[key] += value * sign

  def HMM_train(self, train_file, iterations):
    X = []
    Y_prime = []
    with open(train_file, encoding="utf-8") as file:
      for line in file:
        X_line = []
        Y_line = []
        words = line.strip().split(" ")
        for word in words:
          word = word.split("_")
          X_line.append(word[0])
          Y_line.append(word[1])
        X.append(X_line)
        Y_prime.append(Y_line)
    self.load_model(X, Y_prime)
    for _ in tqdm(range(iterations)):
      for x, y in zip(X, Y_prime):
        y_hat = self.viterbi(x)
        phi_prime = self.create_features(x, y)
        phi_hat = self.create_features(x, y_hat)
        self.update_weight(phi_prime, 1)
        self.update_weight(phi_hat, -1)
  
  def HMM_test(self, test_file):
    with open('/content/myanswer.txt', 'w', encoding="utf-8") as out, open(test_file, encoding="utf-8") as f:
      for line in f:
        x = line.strip().split()
        y = self.viterbi(x)
        print(' '.join(y), file=out)

  def load_model(self, X, Y_prime):
    for word, tag in zip(X, Y_prime):
      for tag1, tag2 in zip(['<s>']+tag, tag+['</s>']):
        self.transition[f'{tag1} {tag2}'] += 1
      self.possible_tags.update(tag)
  
  def create_trans(self, tag1, tag2):
    return [f'T {tag1} {tag2}']

  def create_emit(self, tag, word):
    result = [f'E {tag} {word}']
    if word[0].isupper():
      result.append(f'CAPS {tag}')
    return result

  def viterbi(self, words):
    l = len(words)
    best_score={}
    best_edge={}
    best_score["0 <s>"] = 0
    best_edge["0 <s>"] = "NULL"
    for i in range(l):
      for prev in self.possible_tags:
        for next in self.possible_tags:
          if str(i)+" "+prev in best_score and prev+" "+next in self.transition:
            score = best_score[str(i)+" "+prev] + sum(self.w[key] for key in self.create_trans(prev, next) + self.create_emit(next, words[i]))
            if str(i+1)+" "+next not in best_score or best_score[str(i+1)+" "+next] < score:
              best_score[str(i+1)+" "+next] = score
              best_edge[str(i+1)+" "+next] = str(i)+" "+prev
    for tag in self.possible_tags:
      if tag+" </s>" in self.transition:
        score = best_score[str(l)+" "+tag] + sum(self.w[key] for key in self.create_trans(tag, "</s>"))
        if str(l+1)+" </s>" not in best_score or best_score[str(l+1)+" </s>"] < score:
          best_score[str(l+1)+" </s>"] = score
          best_edge[str(l+1)+" </s>"] = str(l)+" "+tag
    tags = []
    next_edge = best_edge[str(l+1)+" </s>"]
    while next_edge != "0 <s>":
      position, tag = next_edge.split()
      tags.append(tag)
      next_edge = best_edge[next_edge]
    tags.reverse()
    return tags

#演習問題
train_input = "/content/nlptutorial/data/wiki-en-train.norm_pos"
test_input = "/content/nlptutorial/data/wiki-en-test.norm"
hmm = hmm_percep()
hmm.HMM_train(train_input, 5)
hmm.HMM_test(test_input)
'''
Accuracy: 86.92% (3966/4563)

Most common mistakes:
NN --> VBN      66
NN --> NNS      33
JJ --> VBN      30
NN --> VBG      28
NN --> JJ       24
NN --> NNP      21
NNS --> NN      17
JJ --> NNS      16
JJ --> VBG      15
NNS --> VBN     13
'''