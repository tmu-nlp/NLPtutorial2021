from collections import defaultdict
import numpy as np
from tqdm import tqdm

class RecurrentNN:
    def __init__(self):
        self.x_ids = defaultdict(lambda: len(self.x_ids)) #素性のID化
        self.y_ids = defaultdict(lambda: len(self.y_ids)) #品詞のID化
        self.feat_lab = []
        self.net = []
        self.node_n = 64


    def create_one_hot(self, id, size):
        vec = np.zeros(size)
        vec[id] = 1

        return vec


    def init_net(self):
        W_rx = (np.random.rand(self.node_n, len(self.x_ids)) - 0.5) / 5 #-0.1以上0.1未満
        W_rh = (np.random.rand(self.node_n, self.node_n) - 0.5) / 5
        b_r = np.zeros(self.node_n)

        W_oh = (np.random.rand(len(self.y_ids), self.node_n) - 0.5) / 5
        b_o = np.zeros(len(self.y_ids))

        self.net = [W_rx, W_rh, b_r, W_oh, b_o]


    def forward_rnn(self, x):
        W_rx, W_rh, b_r, W_oh, b_o = self.net
        h = [0] * len(x) #各時刻tにおける隠れ層の値
        p = [0] * len(x) #各時刻tにおける出力の確率分布の値
        y = [0] * len(x) #各時刻tにおける出力

        for t in range(len(x)):
            if t > 0:
                h[t] = np.tanh(np.dot(W_rx, x[t]) + np.dot(W_rh, h[t-1]) + b_r)
            else:
                h[t] = np.tanh(np.dot(W_rx, x[t]) + b_r)
            p[t] = np.tanh(np.dot(W_oh, h[t]) + b_o)
            y[t] = np.argmax(p[t])

        return np.array(h), np.array(p), np.array(y)


    def gradient_rnn(self, x, h, p, y_d):
        W_rx, W_rh, b_r, W_oh, b_o = self.net
        d_W_rx = np.zeros(W_rx.shape)
        d_W_rh = np.zeros(W_rh.shape)
        d_b_r = np.zeros(b_r.shape)

        d_W_oh = np.zeros(W_oh.shape)
        d_b_o = np.zeros(b_o.shape)

        d_r_d = np.zeros(len(b_r)) #次の時間から伝播するエラー

        for t in range(len(x)-1, -1, -1):
            p_d = self.create_one_hot(y_d[t], len(self.y_ids))
            d_o_d = p_d - p[t] #出力層エラー
            d_W_oh += np.outer(d_o_d, h[t]); d_b_o += d_o_d #出力層重み勾配
            d_r = np.dot(d_r_d, W_rh) + np.dot(d_o_d, W_oh) #逆伝播
            d_r_d = d_r * (1 - h[t]**2) #tanhの勾配
            d_W_rx += np.outer(d_r_d, x[t]); d_b_r += d_r_d #隠れ層重み勾配
            if t != 0:
                d_W_rh += np.outer(d_r_d, h[t-1])
        return d_W_rx, d_W_rh, d_b_r, d_W_oh, d_b_o


    def update_weights(self, delta, l):
        for i in range(5):
            self.net[i] += l * delta[i]


    def train(self, train_file, iter):
        #素性作成
        with open(train_file) as train_f:
            data = []
            for line in train_f:
                X, Y = [], []
                line = line.strip().split()
                for word_pos in line:
                    word, pos = word_pos.split('_')
                    word = word.lower()
                    self.x_ids[word]
                    self.y_ids[pos]
                    X.append(word)
                    Y.append(pos)
                data.append([X, Y])

            for words, poses in data:
                word_vec, pos_vec = [], []
                for word in words:
                    word_vec.append(self.create_one_hot(self.x_ids[word], len(self.x_ids)))
                for pos in poses:
                    pos_vec.append(self.y_ids[pos])
                self.feat_lab.append([np.array(word_vec), np.array(pos_vec)])

        #ネットワーク初期化
        self.init_net()
    
        #学習
        for i in tqdm(range(iter)):
            for x, y_d in self.feat_lab:
                h, p, y = self.forward_rnn(x)
                delta = self.gradient_rnn(x, h, p, y_d)
                self.update_weights(delta, l=0.01)


    def test(self, test_file):
        with open(test_file) as test_f, open('./my_answer.txt', 'w') as ans_f:
            for line in test_f:
                words = line.strip().split()
                vec = []
                for word in words:
                    if word in self.x_ids.keys():
                        vec.append(self.create_one_hot(self.x_ids[word], len(self.x_ids)))
                    else:
                        vec.append(np.zeros(len(self.x_ids)))
                
                vec = np.array(vec)
                h, p, y = self.forward_rnn(vec)
                res = []
                id2pos = {self.y_ids[k]: k for k in self.y_ids}
                for pred in y:
                    res.append(str(id2pos[pred]))
                
                ans_f.write(' '.join(res) + '\n')


if __name__ == '__main__':
    RNN = RecurrentNN()
    train_file = '../data/wiki-en-train.norm_pos'
    RNN.train(train_file, iter=5)
    test_file = '../data/wiki-en-test.norm'
    RNN.test(test_file)

'''
RNN(iter=5, node=64, l=0.01)
Accuracy: 80.54% (3675/4563)

Most common mistakes:
JJ --> NN       118
NNP --> NN      84
NNS --> NN      74
RB --> NN       57
-LRB- --> NN    46
DT --> NN       44
-RRB- --> NN    43
VBN --> NN      37
IN --> NN       34
VBP --> NN      27


HMM
Accuracy: 90.75% (4141/4563)

Most common mistakes:
NNS --> NN      45
NN --> JJ       27
JJ --> DT       22
NNP --> NN      22
JJ --> NN       12
VBN --> NN      12
NN --> IN       11
NN --> DT       10
NNP --> JJ      8
VBP --> VB      7
'''
