import numpy as np
from collections import defaultdict
import tqdm

def create_features(x, ids):
    phi = [0] * len(ids)
    words = x.split(' ')
    for word in words:
        phi[ids['UNI:' + word]] += 1
    return phi

def forward_nn(net, phi0):
    phi = [0] * (len(net) + 1)
    phi[0] = phi0 #各層の値
    for i in range(len(net)):
        w, b = net[i]
        #パーセプトロン予測
        phi[i + 1] = np.tanh(np.dot(w, phi[i]) + b).T
    return phi #各層の結果を返す

def backward_nn(net, phi, yd):
    j = len(net)
    delta = np.zeros(j + 1, dtype=np.ndarray) 
    delta[-1] = np.array([float(yd) - phi[j][0]])
    deltad = np.zeros(j + 1, dtype=np.ndarray)
    for i in range(j, 0, -1):
        deltad[i] = delta[i] * (1 - phi[i]**2).T
        w, b = net[i - 1]
        delta[i - 1] = np.dot(deltad[i], w)
    return deltad

def update_weights(net, phi, deltad, l):
    for i in range(len(net)):
        w, b = net[i]
        w += l * np.outer(deltad[i + 1], phi[i])
        b += l * deltad[i + 1]

def NN(dataname, iteration, l, node, layer):
    ids = defaultdict(lambda: len(ids))
    feat_lab = []
    
    with open(dataname, encoding="utf-8") as data:
        for line in data:
            y, x = line.strip().split('\t')
            for word in x.split(' '):
                ids['UNI:' + word]

    with open(dataname, encoding="utf-8") as data:
        for line in data:
            y, x = line.strip().split('\t') 
            feat_lab.append((create_features(x, ids), y)) 
    
    #initialize net ramdomly
    net = []

    # 入力層
    w_i = np.random.rand(node, len(ids)) - 0.5
    b_i = np.random.rand(1, node)
    net.append((w_i, b_i))

    # 隠れ層
    while len(net) < layer:
        w = np.random.rand(node, node) - 0.5
        b = np.random.rand(1, node)
        net.append((w, b))

    # 出力層
    w_o = np.random.rand(1, node) - 0.5
    b_o = np.random.rand(1, 1)
    net.append((w_o, b_o))
    
    for i in range(iteration):
        for phi0, y in feat_lab:
            phi = forward_nn(net, phi0)
            deltad = backward_nn(net, phi, y)
            update_weights(net, phi, deltad, l)
    
    return net, ids



def create_features_test(line, ids):
    phi = [0] * len(ids)
    words = line.split(' ')
    for word in words:
        if 'UNI:' + word in ids:
            phi[ids['UNI:' + word]] += 1
    return phi

def predict_one(net, phi0):
    phi = [0] * (len(net) + 1)
    phi[0] = phi0
    for i in range(len(net)):
        w, b = net[i]
        phi[i + 1] = np.tanh(np.dot(w, phi[i]) + b).T
    score = phi[len(net)][0]

    if score >= 0:
        return 1
    else:
        return -1 

def predict_all(net, ids, file_name):
    input_file = open(file_name, 'r')
    ans = open('07-answer.labeled', 'w')

    for line in input_file:
        line = line.strip()
        phi = create_features_test(line, ids)
        yd = predict_one(net, phi)
        ans.write('{}\t{}\n'.format(yd, line))
    
    input_file.close()
    ans.close()

train = 'titles-en-train.labeled'
test = 'titles-en-test.word'
net, ids = NN(train, 1, 0.1, 2, 1)
predict_all(net, ids, test)

'''

$ python grade-prediction.py titles-en-test.labeled 07-answer.labeled

Accuracy = 91.250443%

'''