# -*- coding: utf-8 -*-
"""Tutorial07.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HMV-Ijll8xZvC4vCA9LOrctTDQnEVcpL
"""

!git clone https://github.com/neubig/nlptutorial.git #gitからCloneする

from collections import defaultdict
import numpy as np

def predict_one(net, phiz):
  phi = [0 for _ in range(len(net)+1)]
  phi[0] = phiz
  for i in range(len(net)):
    w, b = net[i]
    phi[i+1] = np.tanh(np.dot(w,phi[i])+b)
  return (1 if phi[len(net)][0] >= 0 else -1)

def create_features(x, ids):
  phi = np.zeros(len(ids), dtype=np.float64)
  words = x.split(" ")
  for word in words:
    phi[ids["UNI:"+word]] +=1
  return phi

def create_features_test(x, ids):
  phi = np.zeros(len(ids), dtype=np.float64)
  words = x.split(" ")
  for word in words:
    if "UNI:"+word in ids:
      phi[ids["UNI:"+word]] +=1
  return phi

def update_weights(net, phi, deltad, lambd):
  for i in range(len(net)-1):
    w, b = net[i]
    w += lambd*np.outer(deltad[i+1], phi[i])
    b += lambd*deltad[i+1]

def forward_nn(net, phiz):
  phi = [0 for _ in range(len(net)+1)]
  phi[0] = phiz
  for i in range(len(net)):
    w, b = net[i]
    phi[i+1] = np.tanh(np.dot(w,phi[i])+b)
  return phi

def backward_nn(net, phi, yd):
  J = len(net)
  delta = np.zeros(J+1, dtype=np.ndarray)
  delta[-1] = np.array([float(yd)-phi[J][0]])
  deltad = np.zeros(J+1, dtype=np.ndarray)
  for i in reversed(range(J)):
    deltad[i+1] = delta[i+1]*(1-phi[i+1]**2)
    w, b = net[i]
    delta[i] = np.dot(deltad[i+1], w)
  return deltad

def predict_all(net, ids, input_file):
  with open("/content/my_answer.txt", 'w', encoding="utf-8") as f:
    with open(input_file, encoding="utf-8") as input:
      for x in input:
        x = x.strip()
        phi = create_features_test(x, ids)
        yd = predict_one(net, phi)
        f.write(str(yd)+"\t"+x+"\n")

def network(iteration, lambd, init_val, layer, layer_node, data_name):
  ids = defaultdict(lambda: len(ids))
  feat_lab = []
  with open(data_name, encoding="utf-8") as data:
    for line in data:
      y,x = line.strip().split('\t')
      for word in x.split(" "):
        ids["UNI:"+word]
  with open(data_name, encoding="utf-8") as data:
    for line in data:
      y,x = line.strip().split('\t')
      feat_lab.append((create_features(x, ids), y))
  net = []
  w = 2 * init_val * np.random.rand(layer_node, len(ids)) - init_val
  b = 2 * init_val * np.random.rand(layer_node) - init_val
  net.append((w, b))
  for _ in range(layer):
    w = 2 * init_val * np.random.rand(layer_node, layer_node) - init_val
    b = 2 * init_val * np.random.rand(layer_node) - init_val
    net.append((w,b))
  w = 2 * init_val * np.random.rand(1, layer_node) - init_val
  b = 2 * init_val * np.random.rand(1) - init_val
  net.append((w,b))
  for i in range(iteration):
    for phiz, y in feat_lab:
      phi = forward_nn(net, phiz)
      deltad = backward_nn(net, phi, y)
      update_weights(net, phi, deltad, lambd)
  return net, ids

#演習問題1
train_input = '/content/nlptutorial/test/03-train-input.txt'
net, ids = network(1,train_input)

#演習問題2
for iteration in [1, 5, 10]:
  for lambd in [0.001, 0.01, 0.1]:
    for init_val in [0.1, 0.5, 1]:
      for layer in [2, 3, 5]:
        for layer_node in [2, 3, 5]:
          net, ids = network(iteration, lambd, init_val, layer, layer_node, '/content/nlptutorial/data/titles-en-train.labeled')
          predict_all(net, ids, '/content/nlptutorial/data/titles-en-test.word')
          print("iteration = {}, lambda = {}, init_val = {}, layer = {}, layer_node = {}  --> {} ".format(iteration, lambd, init_val, layer, layer_node, calc_accuracy()))

'''
iteration = 1のときは, lambda = 0.1ぐらいが安定して高い, init_valはあまり精度に関係なさそう, layer_nodeは大きいほど良いが, layerの大きさは精度にばらつきがある
layer = 2, layer_node = 5ぐらいが安定して高い
iteration = 1, lambda = 0.001, init_val = 0.1, layer = 2, layer_node = 2  --> Accuracy = 52.320227% 
iteration = 1, lambda = 0.01, init_val = 0.1, layer = 2, layer_node = 2  --> Accuracy = 52.320227% 
iteration = 1, lambda = 0.1, init_val = 0.1, layer = 2, layer_node = 2  --> Accuracy = 89.727241% 
iteration = 1, lambda = 0.1, init_val = 0.1, layer = 2, layer_node = 3  --> Accuracy = 90.294013% 
iteration = 1, lambda = 0.1, init_val = 0.1, layer = 2, layer_node = 5  --> Accuracy = 91.037903% 
iteration = 1, lambda = 0.1, init_val = 0.1, layer = 3, layer_node = 5  --> Accuracy = 87.814382% 
iteration = 1, lambda = 0.1, init_val = 0.1, layer = 5, layer_node = 5  --> Accuracy = 52.320227% 
iteration = 1, lambda = 0.1, init_val = 0.5, layer = 2, layer_node = 2  --> Accuracy = 47.679773% 
iteration = 1, lambda = 0.1, init_val = 0.5, layer = 2, layer_node = 3  --> Accuracy = 89.514701% 
iteration = 1, lambda = 0.1, init_val = 0.5, layer = 2, layer_node = 5  --> Accuracy = 90.435707% 
iteration = 1, lambda = 0.1, init_val = 0.5, layer = 3, layer_node = 2  --> Accuracy = 88.239462% 
iteration = 1, lambda = 0.1, init_val = 0.5, layer = 3, layer_node = 3  --> Accuracy = 87.318456% 
iteration = 1, lambda = 0.1, init_val = 0.5, layer = 3, layer_node = 5  --> Accuracy = 91.250443% 
iteration = 1, lambda = 0.1, init_val = 0.5, layer = 5, layer_node = 2  --> Accuracy = 52.320227% 
iteration = 1, lambda = 0.1, init_val = 0.5, layer = 5, layer_node = 3  --> Accuracy = 82.748849% 
iteration = 1, lambda = 0.1, init_val = 0.5, layer = 5, layer_node = 5  --> Accuracy = 88.629118% 
iteration = 1, lambda = 0.1, init_val = 1, layer = 2, layer_node = 5  --> Accuracy = 88.026922% 

iteration = 5のときは, lambda = 0.1のときが精度が安定して高い, 同じくlayer=2, layer_node=5がinit_val関係なく高い
iteration = 5, lambda = 0.1, init_val = 0.1, layer = 2, layer_node = 2  --> Accuracy = 89.373007% 
iteration = 5, lambda = 0.1, init_val = 0.1, layer = 2, layer_node = 3  --> Accuracy = 91.746369% 
iteration = 5, lambda = 0.1, init_val = 0.1, layer = 2, layer_node = 5  --> Accuracy = 92.313142% 
iteration = 5, lambda = 0.1, init_val = 0.5, layer = 2, layer_node = 2  --> Accuracy = 52.320227% 
iteration = 5, lambda = 0.1, init_val = 0.5, layer = 2, layer_node = 3  --> Accuracy = 89.266738% 
iteration = 5, lambda = 0.1, init_val = 0.5, layer = 2, layer_node = 5  --> Accuracy = 93.127878% 
iteration = 5, lambda = 0.1, init_val = 1, layer = 2, layer_node = 2  --> Accuracy = 91.179596% 
iteration = 5, lambda = 0.1, init_val = 1, layer = 2, layer_node = 3  --> Accuracy = 90.294013% 
iteration = 5, lambda = 0.1, init_val = 1, layer = 2, layer_node = 5  --> Accuracy = 92.206872%
'''

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