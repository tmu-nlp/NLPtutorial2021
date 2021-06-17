import numpy as np
import dill
from collections import *
iteration = 1
hidden_layers = 1
hidden_node = 2
Lamda = 0.1




def forward_nn(network, phi_in):
    phi_all = [phi_in]
    for i in range(len(network)):
        w, b = network[i]
        #print(w.shape)
        #前の層の値に基づいて値を計算
        #print(phi_all[i])
        phi_all.append(np.tanh(np.dot(w, phi_all[i]) + b))

    return phi_all


def backward_nn(network, phi_all, y):
    J = len(network)
    delta = [np.ndarray for _ in range(J)]
    delta.append(y - phi_all[J-1])
    d_delta = [np.ndarray for _ in range(J + 1)]
    for i in reversed(range(J)):
        d_delta[i+1] = delta[i+1] * (1- phi_all[i+1]**2)
        w, b = network[i]
        
        #print(w)
        #print(d_delta[i+1])
        delta[i] = np.dot(d_delta[i+1], w)

    return d_delta


def update_weights(network, phi_all, d_delta, Lamda):
    for i in range(len(network)):
        w, b = network[i]
        w += Lamda * np.outer(d_delta[i+1], phi_all[i])
        b += Lamda * d_delta[i+1]
    return 0


def init_net(ids):
    net = []

    #入力層、隠れ層、出力層それぞれ-0.5から0.5の間で重みを初期化
    w_in = np.random.rand(hidden_node, len(ids)) - 0.5
    b_in = np.random.rand(hidden_node) - 0.5
    net.append((w_in, b_in))

    for i in range(hidden_layers):
        w_h = np.random.rand(hidden_node, hidden_node) - 0.5
        b_h = np.random.rand(hidden_node) - 0.5
        net.append((w_h, b_h))
    
    w_out = np.random.rand(1, hidden_node) - 0.5
    b_out = np.random.rand(1) - 0.5
    net.append((w_out, b_out))

    return net


def create_features(sen, ids):
    phi = np.zeros(len(ids))
    words = sen.split()
    for word in words:
        phi[ids[f'UNI:{word}']] += 1

    return phi


def train_nn(train_file):
    ids = defaultdict(lambda: len(ids))
    feat_lab = []
    train = []
    # 素性を作り、ネットワークをランダムな値で初期化
    train_data = ''
    with open(train_file) as f:
        for line in f:
            #print(line.strip().split('\t'))
            y, sen = line.strip().split('\t')
            y = int(y)
            train.append((sen, y))
            for word in sen.split(' '):
                ids[f'UNI:{word}']
                
    for sen, y in train:
            feat_lab.append((create_features(sen, ids), y))

    network = init_net(ids)

    for i in range(iteration):
        error = 0
        for one in feat_lab:
            phi_in, y = one
            phi = forward_nn(network, phi_in)
            d_delta = backward_nn(network, phi, y)
            update_weights(network, phi, d_delta, Lamda)
            error += abs(y - phi[len(network)][0])
    return network, ids


##推論
def forward_nn_predict(network, phi_in):
    phi_all = [phi_in]
    for i in range(len(network)):
        w, b = network[i]
        phi_all.append(np.tanh(np.dot(w, phi_all[i]) + b))
    return 1 if phi_all[len(network) - 1] >= 0 else -1


def test_nn(test_file, to_ans_file, network, ids):
    with open(test_file) as f, open(to_ans_file, 'w') as write_ans:
        for line in f:
            line = line.strip()
            phi_in = create_features(line, ids)
            #1 or -1
            predict_score = forward_nn_predict(network, phi_in)
            print(predict_score, file=write_ans)


if __name__ == '__main__':
    train_file = './nlptutorial/data/titles-en-train.labeled'
    test_file = './nlptutorial/data/titles-en-test.word'
    to_ans_file = 'my_ans.txt'

    network, ids = train_nn(train_file)
    test_nn(test_file, to_ans_file, network, ids)





    




