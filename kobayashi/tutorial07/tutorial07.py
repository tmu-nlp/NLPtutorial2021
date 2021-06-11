import numpy as np
from collections import defaultdict
import pickle

class NeuralNet:
    def __init__(self, hidden_n, node_n):
        self.hidden_n = hidden_n
        self.node_n = node_n
        self.ids = defaultdict(lambda: len(self.ids))
        self.feat_lab = [] #feature, label

    def create_features(self, x):
        phi = [0 for _ in range(len(self.ids))]
        words = x.strip().split()
        for word in words:
            phi[self.ids["UNI:" + word]] += 1
        return phi

    def network(self): #ランダムに重みの初期値を設定(重みを均一に更新させないため)
        net = []

        #input layer
        W_i = np.random.rand(self.node_n, len(self.ids)) - 0.5 #-0.5以上0.5以下
        b_i = np.zeros(self.node_n)
        net.append((W_i, b_i))

        #hidden layer
        for _ in range(self.hidden_n):
            w = np.random.rand(self.node_n, self.node_n) - 0.5
            b = np.zeros(self.node_n)
            net.append((w, b))

        #output layer
        W_o = np.random.rand(1, self.node_n) - 0.5
        b_o = np.zeros(1)
        net.append((W_o, b_o))

        return net

    def forward_nn(self, net, phi_0):
        phi = [0 for _ in range(len(net)+1)]
        phi[0] = phi_0
        for i in range(len(net)):
            w, b = net[i]
            phi[i+1] = np.tanh(np.dot(w, phi[i]) + b)
        return phi
        
    def backward_nn(self, net, phi, y_d):
        J = len(net)
        delta = [0 for _ in range(J)]
        delta.append(y_d - phi[J])
        delta_d = [0 for _ in range(J+1)]
        for i in range(J-1, -1, -1):
            delta_d[i+1] = delta[i+1] * (1 - phi[i+1]**2)
            w, b = net[i]
            delta[i] = np.dot(delta_d[i+1], w)
        return delta_d

    def update_weights(self, net, phi, delta_d, l):
        for i in range(len(net)):
            w, b = net[i]
            w +=  l * np.outer(delta_d[i+1], phi[i])
            b += l * delta_d[i+1]

    def train(self, train_file, iter, l):
        data = []
        with open(train_file) as train_f:
            for line in train_f:
                y, x = line.strip().split('\t')
                for word in x.split():
                    self.ids['UNI:' + word]
                data.append([x, y])
        for x, y in data:
            self.feat_lab.append([self.create_features(x), y])

        net = self.network()

        for i in range(iter):
            for phi_0, y in self.feat_lab:
                phi = self.forward_nn(net, phi_0)
                delta_d = self.backward_nn(net, phi, int(y))
                self.update_weights(net, phi, delta_d, l)

        with open('weight.pkl', 'wb') as weight_f:
            pickle.dump(net, weight_f)

        with open('id.txt', 'w') as id_f:
            for w, c in self.ids.items():
                id_f.write(f'{w}\t{c}\n')
            

    def create_features_test(self, x):
        phi = [0 for _ in range(len(self.ids))]
        words = x.split()
        for word in words:
            if "UNI:" + word in self.ids: #idsに含まれる場合のみ
                phi[self.ids["UNI:" + word]] += 1
        return phi

    def predict_one(self, net, phi_0):
        phi = [0 for _ in range(len(net)+1)]
        phi[0] = phi_0
        for i in range(len(net)):
            w, b = net[i]
            phi[i+1] = np.tanh(np.dot(w, phi[i]) + b)
        score = phi[len(net)][0]
        
        if score >= 0:
            return 1
        return -1

    def test(self, test_file):
        with open('weight.pkl', 'rb') as weight_f, open('id.txt') as id, \
            open(test_file) as test_f, open('my_answer.txt', 'w') as ans_f:
            net = pickle.load(weight_f)
            ids = defaultdict(lambda: 0)
            for line in id:
                w, c = line.split('\t')
                ids[w] = c

            for x in test_f:
                phi = self.create_features_test(x)
                pred = self.predict_one(net, phi)
                ans_f.write(f'{pred}\t{x}')

if __name__ == '__main__':
    '''
    #演習課題1
    NN = NeuralNet(1, 2)
    train_file = '../test/03-train-input.txt'
    NN.train(train_file, iter=1, l=0.1)
    '''

    #演習課題2
    NN = NeuralNet(1, 2)
    train_file = '../data/titles-en-train.labeled'
    NN.train(train_file, iter=1, l=0.1)
    test_file = '../data/titles-en-test.word'
    NN.test(test_file)

'''
結果
Accuracy = 89.798087% (hidden=1, node=2, iter=1, l=0.1)
Accuracy = 90.152320% (hidden=1, node=2, iter=3, l=0.1)
Accuracy = 93.269571% (hidden=1, node=2, iter=5, l=0.1)
Accuracy = 87.920652% (hidden=3, node=2, iter=1, l=0.1)
Accuracy = 52.320227% (hidden=5, node=2, iter=1, l=0.1)
Accuracy = 83.917818% (hidden=1, node=4, iter=5, l=0.1)
'''