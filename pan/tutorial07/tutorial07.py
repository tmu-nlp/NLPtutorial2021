import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from typing import List


# 学習1回、隠れ層1つ、隠れ層のノード2つ
# networkの初期化
# 関数定義
class NeuralNet():
    def __init__(self, layer_num, node_num):
        self.L = layer_num
        self.N = node_num
        # 素性を整数IDに変換する(素性のID化)
        self.ids = defaultdict(lambda: len(self.ids))
        self.feat_lab = []
        self.net = []

    def prepare(self, input_file):
        # 辞書作成
        input_data = []
        with open(input_file) as f:
            for line in f:
                # '\t'で分割
                # delete the space
                label, sentence = line.rstrip().split('\t')
                for word in sentence.split():
                    self.ids['UNI:' + word]
                input_data.append([sentence, int(label)])
        for sentence, label in input_data:
            self.feat_lab.append([self.create_features(sentence), label])

    # 素性の初期化(pdf)
    # 初期値は[-0.1, 0.1]でランダムに初期化
    def init_net(self):
        # 入力層
        w_in = np.random.rand(self.N, len(self.ids)) / 5 - 0.1
        b_in = np.random.rand(self.N) / 5 - 0.1
        self.net.append((w_in, b_in))

        # 隠れ層
        for _ in range(self.L - 1):
            w = np.random.rand(self.N, self.N) / 5 - 0.1
            b = np.random.rand(self.N) / 5 - 0.1
            self.net.append((w, b))

        # 出力層
        w_out = np.random.rand(1, self.N) / 5 - 0.1
        b_out = np.random.rand(1) / 5 - 0.1
        self.net.append((w_out, b_out))

    # 学習を行う
    def train(self, num_iter=10):
        for i in tqdm(range(num_iter)):
            for phi_0, y in tqdm(self.feat_lab):
                phi = self.forward_nn(phi_0)
                delta_d = self.backward_nn(phi, y)
                self.update_weights(phi, delta_d, 0.1)
        return self.net, self.ids

    def test(self, sentence):
        phi0 = self.create_features(sentence.rstrip(), False)
        score = self.predict_one(phi0)
        return 1 if score > 0 else -1

    # パーセプトロンの予測
    def predict_one(self, phi0):
        phis = [phi0]
        for i in range(len(self.net)):
            w, b = self.net[i]
            phis.append(np.tanh(np.dot(w, phis[i]) + b))
        return phis[len(self.net)][0]

    # 素性のID化：素性を整数IDに変換する（疑似コード）
    def create_features(self, sentence, is_train = True):
        phi = [0 for _ in range(len(self.ids))]
        if is_train:
            for word in sentence.split():
                key = 'UNI:' + word
                phi[self.ids[key]] += 1
        else:
            for word in sentence.split():
                key = 'UNI:' + word
                if key in self.ids:
                    phi[self.ids[key]] += 1
        return phi

    # ニューラルネットの伝搬コード
    def forward_nn(self, phi_0):
        # 各層への入力ベクトル
        phi = [0 for _ in range(len(self.net)+1)]
        phi[0] = phi_0
        for i in range(len(self.net)):
            w, b = self.net[i]
            # 前の層の値に基づいて値を計算
            phi[i+1] = np.tanh(np.dot(w, phi[i]) + b)
        # 各層の結果を返す
        return phi

    # 逆伝搬のコード
    def backward_nn(self, phi, y_d):
        J = len(self.net)
        # J+1個の配列
        delta = [0 for _ in range(J)]
        delta.append(y_d - phi[J])
        delta_d = [0 for _ in range(J+1)]
        for i in reversed(range(J)):
            delta_d[i+1] = delta[i+1] * (1 - phi[i+1] ** 2)
            w, b = self.net[i]
            delta[i] = np.dot(delta_d[i+1], w)
        return delta_d

    # 重み更新のコード
    def update_weights(self, phi, delta_d, lamb):
        for i in range(len(self.net)):
            w, b = self.net[i]
            w += lamb * np.outer(delta_d[i+1], phi[i])
            b += lamb * delta_d[i+1]

def check_score(gold_file: str, pred: List[int], detail: bool = False):
   gold = []
   with open(gold_file, mode = 'r', encoding = 'utf-8') as f:
      for line in f:
         label = int(line.split('\t')[0])
         gold.append(label)
   gold = np.array(gold)
   pred = np.array(pred)
   if detail:
      print(classification_report(gold, pred))
   print(f'accuracy: {accuracy_score(gold, pred)}')

if __name__ == '__main__':
    input_path = '/users/kcnco/github/NLPtutorial2021/pan/tutorial07/titles-en-train.labeled'
    test_path = '/users/kcnco/github/NLPtutorial2021/pan/tutorial07/titles-en-test.word'

    net = NeuralNet(1, 2)
    print('Loading data...')
    net.prepare(input_path)
    net.init_net()
    print('Training...')
    net.train(1)
    print('Predicting...')
    ans = []
    for sentence in open(test_path):
        pred = net.test(sentence)
        ans.append(pred)
    check_score('/users/kcnco/github/NLPtutorial2021/pan/tutorial07/titles-en-test.labeled', ans, True)
