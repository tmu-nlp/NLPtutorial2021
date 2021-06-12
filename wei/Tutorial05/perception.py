from collections import defaultdict

# 1つの事例に対する予測, w-> 重みで、defaultdict(int), phi -> 各k:v(素性:重み)の辞書
def predict_one(w, phi):
    score = 0                   # score = w*φ(x)
    for k,v in phi.items():     # k->'UNI:word';v->counts≒weights
        if k in w:
            score += v * w[k]
    if score >= 0:
        return 1
    else:
        return -1

# 1―gram素性を作成
def create_features(x):             # xは素性リストで、各素性に重みを割り振る
    phi = defaultdict(int)
    for word in x:
        phi['UNI:' + word] += 1
    return phi
    # defaultdict(int, {'UNI:word': counts,...})を返す

    # 重みを更新: w <- w + yφ(x), y -> label
    # y=1の場合、φ(x)の素性の重みを増やす≒「yes」の事例の素性により大きな重み
    # そうでないと、より小さな重みを
def update_weight(w, phi, y):
    for k, v in phi.items():
        w[k] += v * y

# 予測
def predict_all(model_file, input_file):
    weights = defaultdict(int)
    #　modelfile、即ちオンライン学習されて出したdatafileを読み込む
    with open(model_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            #print(line)
            feature = line[0]
            weight = line[1]
            weights[feature] = int(weight)
    print(weights)
    #　testfileを読み込み、
    with open(input_file, 'r', encoding= 'utf-8') as filein, \
        open('./my_answer.txt', 'w', encoding= ' utf-8') as fileout:
        for line in filein.readlines():
            #print(line.strip().split())
            phi = create_features(line.strip().split())
            y_pred = predict_one(weights, phi)
            fileout.write(str(y_pred) + '\t' + line)


class MyPerceptron:
    # ラベル付きデータからオンライン学習を行い、予測結果次第で重みを更新。
    def train_perceptron(self, model_file, output_file, num_iterations: int):
        '''
        model_file = 'titles-en-train.labeled'
        output_file = 'answer.txt'
        '''
        weight = defaultdict(int)
        with open(model_file, 'r', encoding='utf=8') as f:
            for i in range(num_iterations):
                for line in f.readlines():
                    line = line.strip().split()
                    features = line[1:]
                    label = int(line[0])
                    phi = create_features(features)
                    print(phi)
                    y_predict = predict_one(weight, phi)   # 各学習事例を予測して分類

                    if y_predict != label:      # 間違った答えの時、重みを更新
                        update_weight(weight, phi, label)
        print(weight)
        # その後、更新した各k:v(素性:重み)をanswer.txtに保存
        with open(output_file, 'w', encoding='utf-8') as answer:
            for k,v in weight.items():
                answer.write(k + '\t' + str(v) + '\n')


    def test_perceptron(self,model_file, test_file):
        predict_all(model_file, test_file)
        '''
        model_file = 'answer.txt'
        test_file = 'titles-en-test.word'
        '''



if __name__ == '__main__':
    perceptron = MyPerceptron()
    print(perceptron.train_perceptron('../data/titles-en-train.labeled',
                     './answer.txt',10))
    print(perceptron.test_perceptron('./answer.txt','../data/titles-en-test.word'))





