import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score

class RNN():
    def __init__(self, node = 2):
        self.node = node
        self.vocab_ids = defaultdict(lambda: len(self.vocab_ids))
        self.pos_ids = defaultdict(lambda: len(self.pos_ids))
        self.feat_lab = []
        np.random.seed(7)

    def init_net(self, data):
        samples=[]
        with open(data, encoding = 'utf-8') as f:
            for line in f:
                #idsの更新
                samples.append(line)
                self.create_ids(line)
            for line in samples:
                sent = self.create_features(line)
                self.feat_lab.append(sent)

        #初期値は[-0.1, 0.1]
        #入力への重み
        self.w_r_x = np.random.rand(self.node, len(self.vocab_ids))/5 - 0.1
        #隠れ層への重み
        self.w_r_h = np.random.rand(self.node, self.node)/5 - 0.1
        self.b_r = np.random.rand(self.node)/5 - 0.1

        #出力への重み
        self.w_o_h = np.random.rand(len(self.pos_ids), self.node)/5 - 0.1
        self.b_o = np.random.rand(len(self.pos_ids))/5 - 0.1

    def create_ids(self, x):
        words = x.strip().split()
        for word in words:
            x, y = word.split('_')
            self.vocab_ids[x]
            self.pos_ids[y]

    def create_features(self, x):
        sent = []
        words = x.strip().split()
        for word in words:
            x, y = word.split('_')
            x_vec = self.create_one_hot(self.vocab_ids[x], len(self.vocab_ids))
            y_vec = self.create_one_hot(self.pos_ids[y], len(self.pos_ids))
            sent.append([x_vec, y_vec])
        return sent

    def find_max(self, p):
        y = 0
        for i in range(len(p)):
            if p[i] > p[y]:
                y = i
        return y

    def create_one_hot(self, id, size):
        vec = np.zeros(size)
        vec[id] = 1
        return vec

    def softmax(self, x):
        u = np.sum(np.exp(x))
        return np.exp(x)/u

    def forward_rnn(self, sent):
        h = []
        p = []
        y = []
        for t in range(len(sent)):
            x, _ = sent[t]
            if t > 0:
                h.append(np.tanh(np.dot(self.w_r_x, x) + np.dot(self.w_r_h, h[t-1]) + self.b_r))
            else:
                h.append(np.tanh(np.dot(self.w_r_x, x) + self.b_r))
            p.append(self.softmax(np.dot(self.w_o_h, h[t]) + self.b_o))
            y.append(self.find_max(p[t]))
        return h, p, y

    def gradient_rnn(self, sent, p, h):
        w_r_x_prime = np.zeros((self.node, len(self.vocab_ids)))
        w_r_h_prime = np.zeros((self.node, self.node))
        b_r_prime = np.zeros(self.node)
        w_o_h_prime = np.zeros((len(self.pos_ids), self.node))
        b_o_prime = np.zeros(len(self.pos_ids))
        delta_r_prime = np.zeros(self.node)     #伝播されるベクトル
        for t in range(len(sent))[::-1]:
            x, y_prime = sent[t]
            #出力層エラー
            delta_o_prime = y_prime - p[t]
            #出力層重み勾配
            w_o_h_prime += np.outer(delta_o_prime, h[t])
            b_o_prime += delta_o_prime
            #逆伝播
            delta_r = np.dot(delta_r_prime, self.w_r_h) + np.dot(delta_o_prime, self.w_o_h)
            #tanhの勾配
            delta_r_prime = delta_r*(1 - h[t]**2)
            #隠れ層の重み勾配
            w_r_x_prime += np.outer(delta_r_prime, x)
            b_r_prime += delta_r_prime
            if t != 0:
                w_r_h_prime += np.outer(delta_r_prime, h[t-1])
        return [w_r_x_prime, w_r_h_prime, b_r_prime, w_o_h_prime, b_o_prime]

    def update_weights(self, delta, lamb):
        w_r_x_prime, w_r_h_prime, b_r_prime, w_o_h_prime, b_o_prime = delta

        self.w_r_x += lamb*w_r_x_prime
        self.w_r_h += lamb*w_r_h_prime
        self.b_r += lamb*b_r_prime
        self.w_o_h += lamb*w_o_h_prime
        self.b_o += lamb*b_o_prime

    def fit(self, data, lr=0.01, iter=10):
        self.init_net(data)
        for i in tqdm(range(iter)):
            for sent in self.feat_lab:
                h, p, _ = self.forward_rnn(sent)
                delta = self.gradient_rnn(sent, p, h)
                self.update_weights(delta, lr)

            with open('/users/kcnco/github/NLPtutorial2021/pan/tutorial08/weight_file', 'w', encoding = 'utf-8') as wf, \
                    open('/users/kcnco/github/NLPtutorial2021/pan/tutorial08/id_file', 'w', encoding = 'utf-8') as idf:
                ws = ['w_r_x', 'w_r_h', 'b_r', 'w_o_h', 'b_o']
                net = [self.w_r_x, self.w_r_h, self.b_r, self.w_o_h, self.b_o]
                for x in zip(ws, net):
                    wf.write(f'{x[0]}\n{x[1]}\n')
                idf.write('word_id\n')
                for key, value in self.vocab_ids.items():
                    idf.write(f'{value}\t{key}\n')
                idf.write('pos_id\n')
                for key, value in self.pos_ids.items():
                    idf.write(f'{value}\t{key}\n')

    #推論

    def test_create_features(self, x):
        phi = []
        words=x.strip().split()
        for word in words:
            if word in self.vocab_ids:
                phi.append(self.create_one_hot(self.vocab_ids[word], len(self.vocab_ids)))
            else:
                phi.append(np.zeros(len(self.vocab_ids)))
        return phi

    def pre_forward_rnn(self, sent):
        h = []
        p = []
        y = []
        for t in range(len(sent)):
            x = sent[t]
            if t > 0:
                h.append(np.tanh(np.dot(self.w_r_x, x) + np.dot(self.w_r_h, h[t-1]) + self.b_r))
            else:
                h.append(np.tanh(np.dot(self.w_r_x, x) + self.b_r))
            p.append(self.softmax(np.dot(self.w_o_h, h[t]) + self.b_o))
            y.append(self.find_max(p[t]))
        return h, p, y

    def predict(self, data, ans):
        with open(data, encoding = 'utf-8') as f,\
                open(ans, 'w', encoding = 'utf-8') as of:
            for line in f:
                sent = []
                phi = self.test_create_features(line)
                h, p, y = self.pre_forward_rnn(phi)
                for ans in y:
                    for value, id in self.pos_ids.items():
                        if id == ans:
                            sent.append(value)
                of.write(f' '.join(sent)+'\n')

if __name__ == '__main__':
    training_file = '/users/kcnco/github/NLPtutorial2021/pan/tutorial08/wiki-en-train.norm_pos'
    rnn = RNN(node=20)
    rnn.fit(training_file, lr = 0.01 ,iter = 7)
    test_file = '/users/kcnco/github/NLPtutorial2021/pan/tutorial08/wiki-en-test.norm'
    ans_file = '/users/kcnco/github/NLPtutorial2021/pan/tutorial08/my_answer'
    rnn.predict(test_file, ans_file)
