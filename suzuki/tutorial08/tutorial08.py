from collections import defaultdict
import numpy as np
from tqdm import tqdm

def find_max(p): # 確率分布pの中からもっとも確率の大きい要素番号を返す
    y = 0
    for i in range(1, len(p)):
        if p[i] > p[y]:
            y = i
    return y

def create_one_hot(id, size):
    vec = np.zeros(size)
    vec[id] = 1
    return vec

def initialize_net(x_ids, y_ids, node):
    Wrx = (np.random.rand(node, len(x_ids)) - 0.5) / 5
    Wrh = (np.random.rand(node, node) - 0.5) / 5
    Woh = (np.random.rand(len(y_ids), node) - 0.5) / 5
    br = (np.random.rand(node) - 0.5) / 5
    bo = (np.random.rand(len(y_ids)) - 0.5) / 5
    net = (Wrx, Wrh, Woh, br, bo)
    return net

def initialize_delta_net(x_ids, y_ids, node):
    d_Wrx = np.zeros((node, len(x_ids)))
    d_Wrh = np.zeros((node, node))
    d_Woh = np.zeros((len(y_ids), node))
    d_br = np.zeros(node)
    d_bo = np.zeros(len(y_ids))
    d_net = (d_Wrx, d_Wrh, d_Woh, d_br, d_bo)
    return d_net

def forward_rnn(net, x):
    Wrx, Wrh, Woh, br, bo = net
    h = [np.ndarray for _ in range(len(x))]
    p = [np.ndarray for _ in range(len(x))]
    y = [np.ndarray for _ in range(len(x))]
    for t in range(len(x)):
        if t > 0:
            h[t] = np.tanh(np.dot(Wrx, x[t]) + np.dot(Wrh, h[t-1]) + br)
        else:
            h[t] = np.tanh(np.dot(Wrx, x[t]) + br)
        p[t] = np.tanh(np.dot(Woh, h[t]) + bo)
        y[t] = find_max(p[t])
    return h, p, y

def gradient_rnn(net, x, h, p, yd, x_ids, y_ids, node):
    d_net = initialize_delta_net(x_ids, y_ids, node)
    d_Wrx, d_Wrh, d_Woh, d_br, d_bo = d_net
    Wrx, Wrh, Woh, br, bo = net
    delta_r_d = np.zeros(len(br))
    for t in range(len(x) - 1, -1, -1):
        delta_o_d = yd[t] - p[t]
        d_Woh += np.outer(delta_o_d, h[t])
        d_bo += delta_o_d
        if t == len(x) - 1:
            delta_r = np.dot(delta_o_d, Woh)
        else:
            delta_r = np.dot(delta_r_d, Wrh) + np.dot(delta_o_d, Woh)
        delta_r_d = delta_r * (1 - h[t]**2)
        d_Wrx += np.outer(delta_r_d, x[t])
        d_br += delta_r_d
        if t != 0:
            d_Wrh += np.outer(delta_r_d, h[t-1])
    return (d_Wrx, d_Wrh, d_Woh, d_br, d_bo)

def update_weights(net, d_net, lam):
    Wrx, Wrh, Woh, br, bo = net
    d_Wrx, d_Wrh, d_Woh, d_br, d_bo = d_net
    Wrx += lam * d_Wrx
    Wrh += lam * d_Wrh
    br += lam * d_br
    Woh += lam * d_Woh
    bo += lam * d_bo
    return (Wrx, Wrh, Woh, br, bo)

def train_rnn(file_name, iteration, node, lam):
    #x...word, y...tag
    x_ids = defaultdict(lambda: len(x_ids))
    y_ids = defaultdict(lambda: len(y_ids))
    feat_lab = []

    with open(file_name, 'r') as f:
        for line in f:
            l = line.strip().split(' ')
            for x_y in l:
                x, y = x_y.split('_')
                x_ids[x]
                y_ids[y]
    
    print(x_ids)
    print('')
    print(y_ids)

    with open(file_name, 'r') as f:
        for line in f:
            x_for_line = [] # 一文あたりのxのonehotベクトル行列
            y_for_line = [] # 一文あたりのyのonehotベクトル行列
            l = line.strip().split(' ')
            for x_y in l:
                x, y = x_y.split('_')
                x_for_line.append(create_one_hot(x_ids[x], len(x_ids)))
                y_for_line.append(create_one_hot(y_ids[y], len(y_ids)))
            feat_lab.append((x_for_line, y_for_line))
    
    net = initialize_net(x_ids, y_ids, node)

    for i in tqdm(range(iteration)):
        for x, yd in feat_lab: # x:一文あたりのxのonehotベクトル行列, y:一文あたりのyのonehotベクトル行列
            h, p, y = forward_rnn(net, x)
            d_net = gradient_rnn(net, x, h, p, yd, x_ids, y_ids, node)
            net = update_weights(net, d_net, lam)

    return net, x_ids, y_ids

def predict(file_name, net, x_ids, y_ids):
    with open(file_name, 'r') as f, open('my_answer.pos', 'w') as ans:
        for line in f:
            l = line.strip().split(' ')
            x_list = []
            for x in l:
                if x in x_ids:
                    x_list.append(create_one_hot(x_ids[x], len(x_ids)))
                else:
                    x_list.append(np.zeros(len(x_ids)))
            h, p, y_list = forward_rnn(net, x_list)
            print(y_list)
            for y in y_list:
                for key, value in y_ids.items():
                    if value == y:
                        print(key, end = ' ', file = ans)
            print(file = ans)


if __name__ == '__main__':
    train_file = 'wiki-en-train.norm_pos'
    test_file = 'wiki-en-test.norm'
    iteration = 40
    node = 100
    lam = 0.01

    net, x_ids, y_ids = train_rnn(train_file, iteration, node, lam)
    predict(test_file, net, x_ids, y_ids)


'''

$ perl gradepos.pl wiki-en-test.pos my_answer.pos
低い。。
Accuracy: 18.76% (856/4563)

Most common mistakes:
IN --> NN       502
JJ --> NN       394
NNS --> NN      391
DT --> NN       363
, --> NN        233
RB --> NN       181
. --> NN        171
VBN --> NN      161
CC --> NN       128
VBZ --> NN      124

'''