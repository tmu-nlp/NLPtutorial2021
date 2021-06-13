from collections import defaultdict
import numpy as np
from tqdm import tqdm
from typing import List
from sklearn.metrics import classification_report, accuracy_score
# 学習1回、隠れ層1つ、隠れ層のノード2つ
# 中国語の備考は自分の理解のため


class NeuralNet:
    def __init__(self, layer_num, node_num):
        self.L = layer_num
        self.N = node_num
        # 素性のID化(素性を整数IDに変換する)。default_factory为可调用的匿名表达式，
        # 每遇到新的key，就将对应的value初始化为当前字典的元素个数，然后把构建好的键值对添加进字典.
        # 添加k-v时，len(dict)仍为 上一次添加后的长度。故第一次添加时，其value为0
        #　defaultdict(<function__main__.<lambda>()>, {'key':len(ids),...})
        self.ids = defaultdict(lambda: len(self.ids))
        self.features = []
        self.net = []

    def predict_one(self, phi0):
        phis = [phi0]
        for i in range(len(self.net)):
            w, b = self.net[i]
            phis.append(np.tanh(np.dot(w, phis[i]) + b))
        return phis[len(self.net)][0]

    # 1―gram素性を整数IDに変換≒各素性に重みを割り振る
    def create_features(self, sentence, is_train=True):
        phi = [0 for _ in range(len(self.ids))]     # 0要素のリスト
        if is_train:
            for word in sentence.split():
                key = 'UNI:' + word
                phi[self.ids[key]] += 1             # ラベル付きデータの各素性に対応する重みを取得
        else:
            for word in sentence.split():
                key = 'UNI:' + word
                if key in self.ids:
                    phi[self.ids[key]] += 1
        #print(phi)
        return phi


    def prepare(self, input_file):
        input_data = []
        # ラベル付きファイルを読み込み、[句,ラベル]のリストを作成
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                label, sentence = line.rstrip().split('\t')
                for word in sentence.split():
                    self.ids['UNI:' + word]
                    # 将'UNI:' + word 作为key追加到ids字典，value为len(ids),
                    # 第一个ids[key]的value为0
                input_data.append([sentence, int(label)])
        # print(input_data)
        #　[[[ラベル付きデータの各素性に対応する重み],ラベル]]の素性リストを作成
        for sentence, label in input_data:
            self.features.append([self.create_features(sentence), label])


    # 素性の初期化
    def init_net(self):
        # np.random.rand(n, m): 0.0以上, 1.0未満の 一様分布のn行m列，行ごとにm列
        # 返回一个或一组服从“0-1”均匀分布的随机样本值，r.v.取值范围为[0,1)，无参数时返回foalt；
        # 在DL中，可用于生成dropout随机向量
        # 入力層，重みは－0.1以上0.1未満でランダムで初期化
        w_in = np.random.rand(self.N, len(self.ids)) / 5 - 0.1
        b_in = np.random.rand(self.N) / 5 - 0.1
        self.net.append((w_in, b_in))

        # 隠れ層
        for _ in range(self.L-1):
            w = np.random.rand(self.N, self.N)/5 - 0.1
            b = np.random.rand(self.N)/5 - 0.1
            self.net.append((w, b))

        # 出力層
        w_out = np.random.rand(1, self.N)/5 - 0.1
        b_out = np.random.rand(1)/5 - 0.1
        self.net.append((w_out, b_out))


    # 順伝播と逆伝播
    def forward_nn(self, phi_0):
        #　各層の値
        phi = [0 for _ in range(len(self.net)+1)]
        phi[0] = phi_0
        for i in range(len(self.net)):
            w, b = self.net[i]
            #　前の層の値に基づいて値を計算
            phi[i+1] = np.tanh(np.dot(w, phi[i]) + b)
            #print(phi)
        #　各層の結果を返す
        return phi

    def backward_nn(self, phi, y_d):        # y_dは正解
        J = len(self.net)
        delta = [0 for _ in range(J)]
        delta.append(y_d - phi[J])
        delta_d = [0 for _ in range(J+1)]
        for i in reversed(range(J)):
            delta_d[i+1] = delta[i+1] * (1 - phi[i+1] ** 2)
            w, b = self.net[i]
            delta[i] = np.dot(delta_d[i+1], w)
        return delta_d

    # 重みの勾配を計算して、更新
    def update_weights(self, phi, delta_d, lamb):
        for i in range(len(self.net)):
            w, b = self.net[i]
            w += lamb * np.outer(delta_d[i+1], phi[i])
            b += lamb * delta_d[i+1]

    # 学習を行う
    def train_nn(self, num_iter,lamb):
        for i in tqdm(range(num_iter)):
            for phi_0, y in tqdm(self.features):
                phi = self.forward_nn(phi_0)
                delta_d = self.backward_nn(phi, y)
                self.update_weights(phi, delta_d, lamb)
        return self.net, self.ids

    def test_nn(self, sentence):
        phi0 = self.create_features(sentence.rstrip(), False)
        score = self.predict_one(phi0)
        return 1 if score >0 else -1

def check_score(gold_file:str, pred:List[int], detail:bool = False):
    gold = []
    with open(gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            label = int(line.split('\t')[0])
            gold.append(label)
    gold = np.array(gold)
    pred = np.array(pred)
    if detail:
        # classification_report(y_true, y_pred, labels=None,target_names=None,sample_weight=None,digits=2,output_dict=False)
        # 显示主要分类指标，返回每个分类标签的P,R,F1，以及总体微平均值，宏平均值，加权均值
        # labels->list,报告中需评估的类标名称；target_names->list，显示与labels对应的名称；
        # sample_weight-> 1维数组，不同数据点在评估结果中所占权重；digits->指定输出格式的精确度
        # output_dict-> 若Ture，评估结果以dict返回

        print(classification_report(gold, pred, digits=3))
    # accuracy_score(y_true, y_pred) -> 精度
    print(f'accuracy:{accuracy_score(gold, pred)}')


if __name__ == '__main__':
    input_file = '../data/titles-en-train.labeled'
    test_file = '../data/titles-en-test.word'
    test_labeled = '../data/titles-en-test.labeled'

    nn = NeuralNet(1, 2)
    print('Loading data...')
    nn.prepare(input_file)
    nn.init_net()
    print('Training...')
    nn.train_nn(1, 0.1)
    print('Predicting...')
    ans = []
    for sentence in open(test_file, 'r', encoding='utf-8'):
        pred = nn.test_nn(sentence)
        ans.append(pred)
    check_score(test_labeled, ans, True)


'''
precision    recall  f1-score   support

          -1      0.910     0.953     0.931      1477
           1      0.946     0.896     0.920      1346

   micro avg      0.926     0.926     0.926      2823
   macro avg      0.928     0.925     0.926      2823
weighted avg      0.927     0.926     0.926      2823

accuracy:0.9259652851576338              
'''




