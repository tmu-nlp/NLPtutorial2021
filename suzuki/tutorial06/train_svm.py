from collections import *
import os

def create_features(x):
    phi = defaultdict(int)
    words = x.split(' ')
    for word in words:
        #1-gram
        phi["UNI:{}".format(word)] += 1
    return phi

def sign(value):
    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0

def predict_one(w, phi):
    score = 0
    for name, value in phi.items(): #score = w * phi(x)
        if name in w:
            score += value * w[name]
    
    if score >= 0:
        return 1
    else:
        return -1

def svm(iterations, data, margin, c):
    w = defaultdict(lambda: 0)
    
    for i in range(iterations):
        for line in data:
            l = line.strip().split('\t')
            x = l[1]
            y = l[0]
            phi = create_features(x)
            
            val = 0
            for name in phi:
                if name in w:
                    val += phi[name] * w[name] * int(y)

            if val <= margin:
                update_weights(w, phi, y, c)
    
    return w

def update_weights(w, phi, y, c):
    for name, value in w.items():
        if abs(value) < c:
            w[name] = 0
        else:
            w[name] -= sign(value) * c
    
    for name, value in phi.items():
        w[name] += value * int(y)

def getw(w, name, c, iter, last):
    if iter != last[name]:
        c_size = c * (iter - last[name])
        #絶対値 < c -> 0に設定
        if abs(w[name]) <= c_size:
            w[name] = 0
        #値 > 0 -> cを引く，値 < 0 -> cを足す
        else:
            w[name] -= sign(w[name]) * c_size
        last[name] = iter
    return w[name]

def predict_all(w, input_file):
    ans = open('06-answer.labeled', 'w')

    for line in input_file:
        line = line.strip()
        phi = create_features(line)
        yd = predict_one(w, phi)
        ans.write('{}\t{}\n'.format(yd, line))
    
    ans.close()

if __name__ == '__main__':
    model = open('titles-en-train.labeled', 'r')
    test = open('titles-en-test.word', 'r')
    
    iteration = 10
    margin = 20
    c = 0.0001

    w = svm(iteration, model, margin, c)
    predict_all(w, test)
    print('iteration : ' + str(iteration) + ', margin : ' + str(margin) + ', c : ' + str(c))
    os.system('python grade-prediction.py titles-en-test.labeled 06-answer.labeled')
    print('')
    
    model.close()
    test.close()

'''

python grade-prediction.py titles-en-test.labeled 06-answer.labeled

・パーセプトロン
Accuracy = 90.967056%

・svm
iteration : 10, margin : 0, c : 0.0001
Accuracy = 89.656394%

iteration : 10, margin : 0, c : 0.001
Accuracy = 91.250443%

iteration : 10, margin : 0, c : 0.01
Accuracy = 72.688629%

iteration : 10, margin : 0, c : 0.1
Accuracy = 54.941552%

iteration : 10, margin : 0, c : 0
Accuracy = 89.939780%

iteration : 10, margin : 0, c : 1.0
Accuracy = 69.535955%

iteration : 10, margin : 1, c : 0.0001
Accuracy = 91.285866%

iteration : 10, margin : 5, c : 0.0001
Accuracy = 91.179596%

iteration : 10, margin : 10, c : 0.0001
Accuracy = 92.206872%

iteration : 10, margin : 20, c : 0.0001
Accuracy = 93.092455%

'''