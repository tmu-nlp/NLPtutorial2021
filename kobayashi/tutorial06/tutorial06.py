from collections import defaultdict

def create_features(x):
    phi = defaultdict(lambda: 0)
    for word in x:
        phi["UNI:" + word] += 1
    return phi

def predict_mar(w, phi, i, last):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value * getw(w, name, i, last)
    return score

def predict_one(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value * w[name]
    if score >= 0:
        return 1
    else:
        return -1

def sign(value):
    if value > 0:
        return 1
    elif value < 0:
        return -1
    return 0

def update_weights(w, phi, y, c=0.0001):
    for name, value in w.items():
        if abs(value) < c:
            w[name] = 0
        else:
            w[name] -= sign(value) * c
    for name, value in phi.items():
        w[name] += value * y

def getw(w, name, iter, last, c=0.0001):
    if iter != last[name]:
        c_size = c * (iter - last[name])
        if abs(w[name]) <= c_size:
            w[name] = 0
        else:
            w[name] -= sign(w[name]) * c_size
        last[name] = iter
    return w[name]

def predict_all(model_file, input_file):
    weights = defaultdict(lambda: 0)

    with open(model_file) as m_file:
        for line in m_file:
            line = line.strip().split('\t')
            name = line[0]
            w = float(line[1])
            weights[name] = w
    
    with open(input_file) as i_file, open('my_answer.txt', 'w') as o_file:
        for x in i_file:
            phi = create_features(x.strip().split())
            y_pred = predict_one(weights, phi)
            o_file.write(f'{y_pred}\t{x}')

def train_svm(train_file, model_file):
    w = defaultdict(lambda: 0)
    last = defaultdict(lambda: 0)
    for _ in range(10):
        with open(train_file) as t_file:
            for i, line in enumerate(t_file):
                line = line.strip().split()
                y = int(line[0])
                x = line[1:]
                phi = create_features(x)
                val = y * predict_mar(w, phi, i, last)
                margin = 10
                if val <= margin:
                    update_weights(w, phi, y)

    with open(model_file, 'w') as m_file:
        for name, w in w.items():
            m_file.write(f'{name}\t{w}\n')
    
def test_svm(model_file, test_file):
    predict_all(model_file, test_file)


if __name__ == '__main__':
    train_file = '../data/titles-en-train.labeled'
    model_file = 'model.txt'
    train_svm(train_file, model_file)
    test_file = '../data/titles-en-test.word'
    test_svm(model_file, test_file)

"""
比較
Accuracy = 93.446688% (perceptron)
Accuracy = 93.694651% (suport vector machine)
"""