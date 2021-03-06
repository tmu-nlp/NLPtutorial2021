# -*- coding: utf-8 -*-
"""Tutorial06.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HMV-Ijll8xZvC4vCA9LOrctTDQnEVcpL
"""

!git clone https://github.com/neubig/nlptutorial.git #gitからCloneする

from collections import defaultdict
import numpy as np

def predict_one(w, phi):
  score = 0
  for name, value in phi.items():
    if name in w:
      score += value*w[name]
  if score>=0:
    return 1
  else:
    return -1

def create_features(x):
  phi = defaultdict(lambda: 0)
  words = x.split(" ")
  for word in words:
    phi["UNI:"+word] +=1
  return phi

def update_weights(w, phi, y, c):
  for name, value in w.items():
    if abs(value) < c:
      w[name] = 0
    else:
      w[name] -= np.sign(value)*c
  for name, value in phi.items():
    w[name]+=value*int(y)

def predict_all(w, input_file):
  with open("/content/my_answer.txt", 'w', encoding="utf-8") as f:
    with open(input_file, encoding="utf-8") as input:
      for x in input:
        x = x.strip()
        phi = create_features(x)
        yd = predict_one(w, phi)
        f.write(str(yd)+"\t"+x+"\n")

def perceptron(iteration, margin, c, data_name):
  w = defaultdict(lambda: 0)
  for i in range(iteration):
    with open(data_name, encoding="utf-8") as data:
      for line in data:
        y,x = line.strip().split('\t')
        phi = create_features(x)
        val=0
        for name, value in phi.items():
          if name in w:
            val += value*w[name]*int(y)
        if val <= margin:
          update_weights(w, phi, y, c)
  return w

#演習問題
for i in [1, 10]:
  for j in [0, 50, 100]:
    for k in [0.0001, 0.001, 0.01, 0.1, 1.0]:
      w = perceptron(i, j, k,'/content/nlptutorial/data/titles-en-train.labeled')
      predict_all(w, '/content/nlptutorial/data/titles-en-test.word')
      print("iteration = {}, margin = {}, c = {} --> {} ".format(i, j, k, calc_accuracy()))

'''
iteration = 1, margin = 0, c = 0.0001 --> Accuracy = 89.656394% 
iteration = 1, margin = 0, c = 0.001 --> Accuracy = 91.250443% 
iteration = 1, margin = 0, c = 0.01 --> Accuracy = 72.688629% 
iteration = 1, margin = 0, c = 0.1 --> Accuracy = 54.941552% 
iteration = 1, margin = 0, c = 1.0 --> Accuracy = 69.535955% 
iteration = 1, margin = 50, c = 0.0001 --> Accuracy = 92.490259% 
iteration = 1, margin = 50, c = 0.001 --> Accuracy = 92.454835% 
iteration = 1, margin = 50, c = 0.01 --> Accuracy = 90.046050% 
iteration = 1, margin = 50, c = 0.1 --> Accuracy = 65.462274% 
iteration = 1, margin = 50, c = 1.0 --> Accuracy = 51.859724% 
iteration = 1, margin = 100, c = 0.0001 --> Accuracy = 92.525682% 
iteration = 1, margin = 100, c = 0.001 --> Accuracy = 92.171449% 
iteration = 1, margin = 100, c = 0.01 --> Accuracy = 87.743535% 
iteration = 1, margin = 100, c = 0.1 --> Accuracy = 75.947574% 
iteration = 1, margin = 100, c = 1.0 --> Accuracy = 51.859724% 
iteration = 10, margin = 0, c = 0.0001 --> Accuracy = 93.163301% 
iteration = 10, margin = 0, c = 0.001 --> Accuracy = 91.958909% 
iteration = 10, margin = 0, c = 0.01 --> Accuracy = 86.787106% 
iteration = 10, margin = 0, c = 0.1 --> Accuracy = 61.176054% 
iteration = 10, margin = 0, c = 1.0 --> Accuracy = 69.535955% 
iteration = 10, margin = 50, c = 0.0001 --> Accuracy = 93.694651% 
iteration = 10, margin = 50, c = 0.001 --> Accuracy = 93.623804% 
iteration = 10, margin = 50, c = 0.01 --> Accuracy = 91.002480% 
iteration = 10, margin = 50, c = 0.1 --> Accuracy = 65.639391% 
iteration = 10, margin = 50, c = 1.0 --> Accuracy = 51.859724% 
iteration = 10, margin = 100, c = 0.0001 --> Accuracy = 93.234148% 
iteration = 10, margin = 100, c = 0.001 --> Accuracy = 93.482111% 
iteration = 10, margin = 100, c = 0.01 --> Accuracy = 90.010627% 
iteration = 10, margin = 100, c = 0.1 --> Accuracy = 76.762310% 
iteration = 10, margin = 100, c = 1.0 --> Accuracy = 51.859724% 
'''

#!/usr/bin/python
def calc_accuracy():
  import sys
  import math

  ref = list()
  test = list()

  # Load the reference file
  ref_file = open('/content/nlptutorial/data/titles-en-test.labeled', "r", encoding="utf-8")
  for line in ref_file:
      line = line.strip()
      columns = line.split("\t")
      ref.append(int(columns[0]))
  ref_file.close()

  # Load the testing file
  test_file = open(r'/content/my_answer.txt', "r", encoding="utf-8_")
  for line in test_file:
      line = line.strip()
      columns = line.split("\t")
      test.append(int(columns[0]))
  test_file.close()

  # Check to make sure that both are the same length
  if len(test) != len(ref):
      print("Lengths of test (%i) and reference (%i) file don't match" % (len(test), len(ref)))
      sys.exit(1)

  total = 0
  correct = 0
  for i in range(0, len(ref)):
      total += 1
      if ref[i] == test[i]:
          correct += 1

  return("Accuracy = %f%%" % float(float(correct)/float(total)*100.0))