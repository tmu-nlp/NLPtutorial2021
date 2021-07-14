from tqdm import tqdm
from collections import defaultdict
import numpy as np

class RNN():
    def __init__(self, node):
        self.node = node
        self.vocab_ids = defaultdict(lambda:len(self.vocab_ids))
        self.pos_ids = defaultdict(lambda:len(self.pos_ids))
        self.feat_lab = []

    def init_net(self, train_file):
        samples =[]
        with open(train_file, 'r', encoding='utf-8') as inf:
            for line in inf:
                # idsを更新
                samples.append(line)
                self.create_ids(line)
            for line in samples:
                sent = self.create_features(line)
                self.feat_lab.append(sent)     # ベクトルリスト集合

        # 初期値はランダムで[-0.1,0.1)
        # 入力への重み
        self.w_in = np.random.rand(self.node, len(self.vocab_ids))/ 5 - 0.1
        # 隠れ層への重み
        self.w_h = np.random.rand(self.node, self.node)/5 - 0.1
        self.b_h = np.random.rand(self.node)/ 5 - 0.1
        # 出力層への重み
        self.w_out = np.random.rand(len(self.pos_ids), self.node)/5 -0.1
        self.b_out = np.random.rand(len(self.pos_ids))/5 - 0.1

    def create_ids(self, x):
        words = x.strip().split()
        for word in words:
            x, y = word.split('_')
            x.lower()
            self.vocab_ids[x]    #　当時の語彙集合の大きさ若しくは該当単語の位置
            self.pos_ids[y]
        return self.vocab_ids, self.pos_ids

    def create_features(self, x):
        featlab = []
        words= x.strip().split()
        for word in words:
            x, y = word.split('_')
            x.lower()
            x_vec = self.create_one_hot(self.vocab_ids[x], len(self.vocab_ids))
            y_vec = self.create_one_hot(self.pos_ids[y], len(self.pos_ids))
            featlab.append([x_vec, y_vec])   # lineごとに各単語と品詞タグのone-hot vector　listを作って、featlabに追加
        return featlab

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
                h.append(np.tanh(np.dot(self.w_in, x) + np.dot(self.w_h, h[t-1]) + self.b_h))

            else:
                h.append(np.tanh(np.dot(self.w_in, x) + self.b_h))
            p.append(self.softmax(np.dot(self.w_out, h[t]) + self.b_out))
            y.append(self.find_max(p[t]))

        return h, p, y

    def gradient_rnn(self, sent, p, h):
        # 勾配重みを初期化
        dw_in = np.zeros((self.node, len(self.vocab_ids)))
        dw_h = np.zeros((self.node, self.node))
        db_h = np.zeros(self.node)
        dw_out = np.zeros((len(self.pos_ids), self.node))
        db_out = np.zeros(len(self.pos_ids))
        delta_r = np.zeros(self.node)
        for t in reversed(range(len(sent))):
            x, y_ = sent[t]
            # 出力層エラー
            delta_r_out = y_ - p[t]
            # 出力層重み勾配
            dw_out += np.outer(delta_r_out, h[t])
            db_out += delta_r_out
            # 逆伝播
            delta_r_h = np.dot(delta_r, self.w_h) + np.dot(delta_r_out, self.w_out)
            # tanhの勾配
            delta_r = delta_r_h * (1 - h[t]**2)
            # 隠れ層の重み勾配
            dw_in += np.outer(delta_r, x)
            db_h += delta_r
            if t != 0:
                dw_h += np.outer(delta_r, h[t-1])
            dnet = dw_in, dw_h, db_h, dw_out, db_out
        return dnet

    def update_weights(self,net, dnet,lamb):
        # self.w_in, self.w_h, self.b_h, self.w_out, self.b_out = net

        for i in range(len(net)):
            net[i] += lamb * dnet[i]
        return net

    def fit(self, data, lr, iter):
        self.init_net(data)
        for i in tqdm(range(iter)):
            for sent in self.feat_lab:
                h, p, _ =self.forward_rnn(sent)
                dnet = self.gradient_rnn(sent, p, h)
                net = [self.w_in, self.w_h, self.b_h, self.w_out, self.b_out]
                self.update_weights(net, dnet, lr)

                with open('weights.txt', 'w', encoding='utf-8') as wf, \
                    open('ids.txt','w', encoding='utf-8') as idf:
                    ws = ['w_in', 'w_h','b_h', 'w_out', 'b_out']
                    for x in zip(ws, net):
                        wf.write(f'{x[0]}\n{x[1]}\n')
                    idf.write('word_ids\n')
                    for k, v in self.vocab_ids.items():
                        idf.write(f'{v}\t{k}\n')
                    idf.write('pos_ids\n')
                    for k,v in self.pos_ids.items():
                        idf.write(f'{v}\t{k}\n')

    # 推論
    def create_test_features(self, x):
        phi = []
        words = x.strip().split()
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
            if t > 0:
                h.append(np.tanh(np.dot(self.w_in, sent[t]) + np.dot(self.w_h, h[t-1]) + self.b_h))
            else:
                h.append(np.tanh(np.dot(self.w_in, sent[t]) + self.b_h))
            p.append(self.softmax(np.dot(self.w_out, h[t]) + self.b_out))
            y.append(self.find_max(p[t]))
        return h, p, y

    def predict(self, data):
        with open(data,'r', encoding='utf-8') as tf, \
            open('answers.txt','w', encoding='utf-8') as ansf:
            for line in tf:
                sent = []
                phi = self.create_test_features(line)
                h, p, y = self.pre_forward_rnn(phi)
                for ans in y:
                    for value, id in self.pos_ids.items():
                        if id == ans:
                            sent.append(f'{value}\t')
                ansf.write(f''.join(sent)+'\n')

if __name__ == '__main__':
    train_file ='../data/wiki-en-train.norm_pos'
    rnn = RNN(node=20)
    rnn.fit(train_file, lr=0.01, iter=7)
    test_file = '../data/wiki-en-test.norm'
    rnn.predict(test_file)

