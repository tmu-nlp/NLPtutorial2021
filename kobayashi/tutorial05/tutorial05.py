from collections import defaultdict

def create_features(x):
    phi = defaultdict(lambda: 0)
    for word in x:
        phi["UNI:" + word] += 1
    return phi

def predict_one(w, phi):
    score = 0
    for name, value in phi.items():
        if name in w:
            score += value * w[name]
    if score >= 0:
        return 1
    else:
        return -1

def update_weights(w, phi, y):
    for name, value in phi.items():
        w[name] += value * y

def predict_all(model_file, input_file):
    weights = defaultdict(lambda: 0)

    with open(model_file) as m_file:
        for line in m_file:
            line = line.strip().split('\t')
            name = line[0]
            w = int(line[1])
            weights[name] = w
    
    with open(input_file) as i_file, open('my_answer.txt', 'w') as o_file:
        for x in i_file:
            phi = create_features(x.strip().split())
            y_pred = predict_one(weights, phi)
            o_file.write(f'{y_pred}\t{x}')

def train_perceptron(train_file, model_file):
    w = defaultdict(lambda: 0)
    for _ in range(100):
        #ループの中で開く
        with open(train_file) as t_file:
            for line in t_file:
                line = line.strip().split()
                y = int(line[0])
                x = line[1:]
                phi = create_features(x)
                y_pred = predict_one(w, phi)
                if y_pred != y:
                    update_weights(w, phi, y)

    with open(model_file, 'w') as m_file:
        for name, w in w.items():
            m_file.write(f'{name}\t{w}\n')
    
def test_perceptron(model_file, test_file):
    predict_all(model_file, test_file)


if __name__ == '__main__':
    train_file = '../data/titles-en-train.labeled'
    model_file = 'model.txt'
    train_perceptron(train_file, model_file)
    test_file = '../data/titles-en-test.word'
    test_perceptron(model_file, test_file)

"""
iteration: 1 -> Accuracy = 90.967056%
iteration: 5 -> Accuracy = 91.852639%
iteration: 10 -> Accuracy = 93.446688%
iteration: 15 -> Accuracy = 93.057032%
iteration: 100 -> Accuracy = 93.552958%
"""