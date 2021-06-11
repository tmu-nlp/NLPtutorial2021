import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
"""
NLPtutorial2020のDockerfile, requirement.txt, Makefileを使用
make docker-run FILE_NAME=./tutorial07/tutorial07.py
"""

class NN():
    def __init__(self, λ=0.1, node=2, layer=1) -> None:
        self.λ = λ
        self.ids = defaultdict(lambda: len(self.ids)) # {word:id}
        self.node = node
        self.layer = layer
        self.network = []
        self.feat_lab = []

    def load_file(self, file):
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
                _, sentence = line.split("\t")
                for word in sentence.split():
                    self.ids[word]
        for line in lines:
            label, sentence = line.split("\t")
            label = int(label)
            words = sentence.split()
            self.feat_lab.append([self.create_features(words), label])
        return self

    def init_network(self):
        # 入力層(ノードの数, 素性の数)
        w_in = np.random.rand(self.node, len(self.ids)) / 5 - 0.1
        b_in = np.random.rand(self.node) / 5 - 0.1
        self.network.append((w_in, b_in))

        # 隠れ層（隠れ層と隠れ層の間の数分）
        for _ in range(self.layer - 1):
            w = np.random.rand(self.node, self.node) / 5 - 0.1
            b = np.random.rand(self.node) / 5 - 0.1
            self.network.append((w, b))

        # 出力層
        w_out = np.random.rand(1, self.node) / 5 - 0.1
        b_out = np.random.rand(1) / 5 - 0.1
        self.network.append((w_out, b_out))

    def create_features(self, words, is_train=True):
        phi = [0 for _ in range(len(self.ids))]
        if is_train:
            for word in words:
                phi[self.ids[word]] += 1
        else:
            for word in words:
                if word in self.ids:
                    phi[self.ids[word]] += 1
        return phi # self.ids[word]のid(val)がphiのindex

    def forward(self, phi_0):
        phi = [0 for _ in range(len(self.network)+1)]
        phi[0] = phi_0
        for i_net in range(len(self.network)):
            w, b = self.network[i_net]
            phi[i_net+1] = np.tanh(np.dot(w, phi[i_net]) + b)
        return phi

    def backward(self, phi, y_d):
        J = len(self.network)
        delta = np.zeros(J+1, dtype=np.ndarray)
        delta[-1] = np.array([float(y_d) - phi[J]])
        delta_d = np.zeros(J+1, dtype=np.ndarray)
        for i in reversed(range(J)):
            delta_d[i+1] = delta[i+1] * (1 - phi[i+1] ** 2)
            w, _ = self.network[i]
            delta[i] = np.dot(delta_d[i+1], w)
        return delta_d

    def update_weights(self, phi, delta_d):
        for i in range(len(self.network)):
            w, b = self.network[i]
            w += self.λ * np.outer(delta_d[i+1], phi[i])
            b += self.λ * delta_d[i+1][0]
            # self.network[i]内のwとbも更新されている
        return self

    def train(self, iter=3):
        for i in range(iter):
            # [{単語のid（1文ごと）:  回数}, {}, ...,{}]
            for phi_0, label in tqdm(self.feat_lab):
                phi = self.forward(phi_0)
                delta_d = self.backward(phi, label)
                self.update_weights(phi, delta_d)
    
    def predict_one(self, phi_0):
        phis = [phi_0]
        for i in range(len(self.network)):
            w, b = self.network[i]
            phis.append(np.tanh(np.dot(w, phis[i]) + b))
        return phis[len(self.network)][0]

    def test(self, sentence):
        phi0 = self.create_features(sentence.rstrip(), False)
        score = self.predict_one(phi0)
        return 1 if score > 0 else -1

def check_score(gold_file, pred, detail=False):
   true = []
   with open(gold_file, mode='r', encoding='utf-8') as f:
      for line in f:
         label = int(line.split("\t")[0])
         true.append(label)
   true = np.array(true)
   pred = np.array(pred)
   if detail:
      print(classification_report(true, pred))
   print(f"Accuracy: {accuracy_score(true, pred)}")


if __name__ == "__main__":
    # train_file = "data/03-train-input.txt"
    train_file = "data/titles-en-train.labeled"
    test_file = "data/titles-en-test.word"

    # ノードの数
    for node in range(1,8,2):
        model = NN(node=node)
        print("Loading...")
        model.load_file(train_file)
        print("Initializing network...")
        model.init_network()
        print("Training...")
            
        model.train()
        ans = []
        with open(test_file) as f:
            for sentence in f:
                pred = model.test(sentence)
                ans.append(pred)
        print("NODE: ", node)   
        check_score("data/titles-en-test.labeled", ans, True)