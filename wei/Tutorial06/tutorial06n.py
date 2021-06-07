from collections import defaultdict
from math import sin



def predict_margin(w, phi, label):
    score = 0
    for key,value in phi.items():
        if key in w:
            score += value * w[key] * label
    return score

'''
param: str
output: dict
'''
def create_features(x):
    phi = defaultdict(int)

    for word in x:
        phi['UNI:' + word] += 1
    return phi

def predict_one(w, phi):
    score = 0
    for name, value in phi.items():    #{'UNI:The': 1,...}
        if name in w:
            score += value * w[name]
    if score >= 0:
        return 1
    else:
        return -1

def predict_all(model_file, input_file):
    '''
    model_file = 'answer.txt'
    input_file = 'titles-en-test.word'
    '''
    weights = defaultdict(int)

    with open(model_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            feature = line[0]
            weight = line[1]
            weights[feature] = float(weight)

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open('./my_answer.txt', 'w', encoding='utf-8') as outfile:
        for line in infile.readlines():

            phi = create_features(line.strip().split())
            y_pred = predict_one(weights, phi)
            outfile.write(str(y_pred) + '\t' + line)


def update_weights(w, phi, y ,c):
    for name, value in w.items():
        if abs(value) < c:
            w[name] = 0
        else:
            w[name] -= sin(value) * c

    for name, value in phi.items():
        w[name] += value * y

def getw(w, name, c, iter, last):
    if iter != last[name]:
        c_size = c*(iter-last[name])
        if abs(w[name]) <= c_size:
            w[name] = 0
        else:
            w[name] -= sin(w[name]) * c_size
        last[name] = iter
    return w[name]

'''
args: dict A, dict B
output: sum(A[x]*B[x]) -> w*phi'''
def dot_w_phi(w, phi):
    sum = 0
    for key, value in phi.items():
        sum += w[key] * phi[key]
    return sum

'''
args: i(iteration <=> epochs), train_file, output_file, margin, c
output: 
'''
def train_svm(i, model_file, output_file, margin, c):
    '''
    model_file = 'titles-en-trained.labeled'
    output_file = 'answer.txt'
    '''
    w = defaultdict(int)
    with open(model_file , 'r', encoding='utf-8') as f:
        for epoch in range(0, i):
            for line in f.readlines():
                line = line.strip().split()
                features = line[1:]
                label = int(line[0])
                phi = create_features(features)
                val = predict_margin(w, phi, label)

                if val <= margin:
                    update_weights(w, phi, label, c)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for k, v in w.items():
            outfile.write(k + '\t' + str(v) + '\n')

def test_svm(model_file, test_file):
    predict_all(model_file, test_file)



if __name__ == '__main__':
    i = 20
    margin = 10
    c = 0.0001
    print(train_svm(i, '../data/titles-en-train.labeled', './answer.txt', margin, c))
    print(test_svm('./answer.txt', '../data/titles-en-test.word'))

