# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1N5VTrFTy3-pVgvZEDm_rR8jTeonkitJ_
"""

!git clone https://github.com/neubig/nlptutorial.git

import math
from collections import defaultdict
from nltk.tree import Tree
from nltk.treeprettyprinter import TreePrettyPrinter


def Read_Grammar(grammar_file):
  nonterm = []
  preterm = defaultdict(list)
  with open(grammar_file, encoding="utf-8") as f:
    for rule in f:
      lhs, rhs, prob = rule.strip().split("\t")
      rhs_symbols = rhs.split(" ")
      if len(rhs_symbols) == 1:
        preterm[rhs].append((lhs, math.log(float(prob))))
      else:
        nonterm.append((lhs, rhs_symbols[0], rhs_symbols[1], math.log(float(prob))))
  return nonterm, preterm

def Terminals(nonterm, preterm, file_name):
  with open(file_name, encoding="utf-8") as f:
    for line in f:
      words = line.strip().split(" ")
      best_score = defaultdict(lambda: -math.inf)
      best_edge = {}
      for i in range(len(words)):
        for lhs, log_prob in preterm[words[i]]:
          best_score[f'{lhs} {i} {i+1}'] = log_prob
      for j in range(2, len(words)+1):
        for i in range(j-2, -1, -1):
          for k in range(i+1, j):
            for sym, lsym, rsym, logprob in nonterm:
              if best_score[f'{lsym} {i} {k}'] > -math.inf and best_score[f'{rsym} {k} {j}'] > -math.inf:
                my_lp = best_score[f'{lsym} {i} {k}'] + best_score[f'{rsym} {k} {j}'] + logprob
                if my_lp > best_score[f'{sym} {i} {j}']:
                  best_score[f'{sym} {i} {j}'] = my_lp
                  best_edge[f'{sym} {i} {j}'] = (f'{lsym} {i} {k}', f'{rsym} {k} {j}')
      #NLPチュートリアルのコードだとできない（AttributeError: type object 'Tree' has no attribute 'parse')
      tree_line = print_sym(f'S 0 {len(words)}', best_edge, words)
      t = Tree.fromstring(tree_line)
      print(TreePrettyPrinter(t).text())
      print(tree_line)

def print_sym(symij, best_edge, words):
  sym, i, j = symij.split(" ")
  if symij in best_edge:
    return "("+sym+" "+print_sym(best_edge[symij][0], best_edge, words)+ " "+print_sym(best_edge[symij][1], best_edge, words)+")"
  else:
    return "("+sym+" "+words[int(i)]+")"

#テスト
train_input = "/content/nlptutorial/test/08-input.txt"
grammar = "/content/nlptutorial/test/08-grammar.txt"
nonterm, preterm = Read_Grammar(grammar)
tree_line = Terminals(nonterm, preterm, train_input)

'''
S                                     
   _____|___________                           
  |                 VP                        
  |      ___________|____                      
  |     |               VP'                   
  |     |        ________|____                 
  |     |       |             PP              
  |     |       |         ____|___             
  |     |       NP       |        NP          
  |     |    ___|___     |     ___|______      
NP_PRP VBD  DT      NN   IN   DT         NN   
  |     |   |       |    |    |          |     
  i    saw  a      girl with  a      telescope

(S (NP_PRP i) (VP (VBD saw) (VP' (NP (DT a) (NN girl)) (PP (IN with) (NP (DT a) (NN telescope))))))
'''

#演習問題
train_input = "/content/nlptutorial/data/wiki-en-short.tok"
grammar = "/content/nlptutorial/data/wiki-en-test.grammar"
nonterm, preterm = Read_Grammar(grammar)
tree_line = Terminals(nonterm, preterm, train_input)