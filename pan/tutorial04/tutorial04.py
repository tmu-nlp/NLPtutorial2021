import os
from math import log2
from typing import List
from collections import defaultdict
BOS = "<s>"
EOS = "</s>"


class HiddenMarkovModel():
   def __init__(self):
      self.context = defaultdict(lambda: 0)
      self.transition = defaultdict(lambda: 0)
      self.emit = defaultdict(lambda: 0)
      self.possible_tags = defaultdict(lambda: False)
      self.lambda_unk = 0.95
      self.N = 1000000

   def train(self, filename: str, out_filename: str = "Model.txt"):
      assert os.path.exists(filename), f"{filename} does not exist."
      # Load and Train
      with open(filename, encoding='utf-8') as f:
         for line in f:
            line = line.rstrip().split()
            pre_tag = BOS
            self.context[pre_tag] += 1
            for wordtags in line:
               word, cur_tag = wordtags.split("_")
               self.transition[f"{pre_tag}_{cur_tag}"] += 1
               self.context[cur_tag] += 1
               self.emit[f"{cur_tag}_{word}"] += 1
               pre_tag = cur_tag
            self.transition[f"{pre_tag}_{EOS}"] += 1
      # Output
      with open(out_filename, mode="w", encoding= "utf-8") as f:
         for key, value in sorted(self.transition.items()):
            pre_tag, cur_tag = key.split("_")
            prob = value/self.context[pre_tag]
            f.write(f"T {pre_tag} {cur_tag} {prob}\n")
         for key, value in sorted(self.emit.items()):
            tag, word = key.split("_")
            prob = self.lambda_unk * value/self.context[tag] + (1 - self.lambda_unk) / self.N
            f.write(f"E {tag} {word} {prob}\n")

   def load(self, filename: str = "Model.txt"):
      assert os.path.exists(filename), f"{filename} does not exist."
      with open(filename, encoding='utf-8') as f:
         for line in f:
            T, a, b, prob = line.split()
            self.possible_tags[a] = True
            if T == "T":
               pre_tag, cur_tag = a, b
               self.transition[f"{pre_tag}_{cur_tag}"] = float(prob)
            else:
               tag, word = a, b
               self.emit[f"{tag}_{word}"] = float(prob)

   def smooth(self, p):
      return self.lambda_unk * p + (1 - self.lambda_unk) / self.N

   def predict(self, sentence: List[str]):
      # foward
      words = sentence.split()
      l = len(words)
      best_edges = defaultdict(lambda: [0, None]) # [score, edge]
      best_edges[f"0_{BOS}"] = [0, None]
      for i in range(l+1):
         word = words[i] if i < l else EOS
         for pre_tag in self.possible_tags.keys():
            keys = self.possible_tags.keys() if i < l else [EOS]
            for cur_tag in keys:
               if f"{i}_{pre_tag}" in best_edges.keys() and f"{pre_tag}_{cur_tag}" in self.transition.keys():
                  score = best_edges[f"{i}_{pre_tag}"][0] -\
                          log2(self.transition[f"{pre_tag}_{cur_tag}"]) -\
                          log2(self.smooth(self.emit[f"{cur_tag}_{word}"]))
                  if f"{i+1}_{cur_tag}" not in best_edges:
                     best_edges[f"{i+1}_{cur_tag}"] = [score, f"{i}_{pre_tag}"]
                  if best_edges[f"{i+1}_{cur_tag}"][0] > score:
                     best_edges[f"{i+1}_{cur_tag}"] = [score, f"{i}_{pre_tag}"]

      # backward
      tags = []
      next_edge = best_edges[f"{l+1}_{EOS}"][1]
      while next_edge[0] != '0':
         idx, tag = next_edge.split("_")
         idx = int(idx)
         tags.append(tag)
         next_edge = best_edges[next_edge][1]
      return " ".join(list(reversed(tags)))


if __name__ == "__main__":
   Model = HiddenMarkovModel()
   Model.train("/users/kcnco/github/NLPtutorial2021/pan/tutorial04/wiki-en-train.norm_pos")
   # Model.train("../../test/05-train-input.txt")
   Model.load()

   # test
   with open("/users/kcnco/github/NLPtutorial2021/pan/tutorial04/wiki-en-test.norm", encoding='utf-8') as f:
   # with open("../../test/05-test-input.txt", encoding='utf-8') as f:
      for line in f:
         print(Model.predict(line))

"""
Accuracy: 87.07% (3973/4563)

Most common mistakes:
 --> .  168
NNS --> NN      45
NN --> JJ       27
NNP --> NN      22
JJ --> DT       22
VBN --> NN      12
JJ --> NN       12
NN --> IN       11
NN --> DT       10
NNP --> JJ      8
"""
