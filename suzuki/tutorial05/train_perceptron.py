from collections import defaultdict

def create_features(x):
    phi = defaultdict(lambda: 0)
    words = x.split(" ")
    for word in words:
        #"uni: "を追加して1-gramを表す
        phi["UNI:{}".format(word)] += 1

    return phi

def predict_one(w, phi):
    score = 0
    for name in phi: #score = w * phi(x)
        if name in w:
            score += phi[name] * w[name]
    
    if score >= 0:
        return 1
    else:
        return -1

def update_weights(w, phi, y):
    for name in phi:
        w[name] += phi[name] * y

def train_perceptron(iteration, target):
    w = defaultdict(lambda: 0)

    for i in range(iteration):
        for line in target:
            line = line.strip()
            xy = line.split("\t") #<y> <x>
            phi = create_features(xy[1])
            yd = predict_one(w, phi)

            if yd != int(xy[0]):
                update_weights(w, phi, int(xy[0]))

    return w

def predict_all(w, input_file):
    ans = open('05-answer.labeled', 'w')

    for line in input_file:
        line = line.strip()
        phi = create_features(line)
        yd = predict_one(w, phi)
        ans.write('{}\t{}\n'.format(yd, line))
    
    ans.close()


if __name__ == '__main__':
    #テスト
    f1 = open('03-train-input.txt', 'r')
    f2 = open('05-model.txt', 'w')
    weights = sorted(train_perceptron(5,f1).items(), key = lambda x: x[0])
    for key, value in weights:
        f2.write("{}\t{:.6f}\n".format(key, float(value)))
    f1.close()
    f2.close()