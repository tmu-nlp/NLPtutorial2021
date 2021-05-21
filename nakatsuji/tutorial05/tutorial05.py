from collections import *


def create_features(x):
    phi = defaultdict(int)
    words = x.split(' ')
    for word in words:
        phi['UNI:' + word] += 1

    return phi

def predict_one(weight, phi):
    score = 0
    for name, value in phi.items():
        if name in weight: score += value * weight[name]

    if score >= 0:
        return 1
    return -1

def predict_all(w, input_file, out_file):
    with open(input_file) as f, open(out_file, 'w') as out:
        for line in f:
            phi = create_features(line)
            y_pred = predict_one(w, phi)
            out.write(f'{y_pred}\t{line}')



def online_learning(iteration_number, data):
    weights = defaultdict(int)

    for _ in range(iteration_number):
        with open(data) as f:
            for line in f:
                #print(line.strip().split('\t'))
                y, x = line.strip().split('\t')
                y = int(y)
                phi = create_features(x)
                pred_y = predict_one(weights, phi)

                if pred_y != y:
                    update_weights(weights, phi, y)
    return weights

def update_weights(weight, phi, y):
    for name, value in phi.items():
        weight[name] += float(value * y)


if __name__ == '__main__':
    path =  '/Users/michitaka/lab/NLP_tutorial/nlptutorial/'

    #テスト
    input = path + 'test/03-train-input.txt'
    answer = path + 'test/03-train-answer.txt'
    #w = online_learning(5, input)
    #print(w)
    #with open('my_weights.txt','w') as f:
    #   for name, value in sorted(w.items()):
    #        f.write(f'{name}\t{value}\n')
        
    #演習
    data = path + 'data/titles-en-train.labeled'
    input = path + 'data/titles-en-test.word'
    weights = online_learning(10, data)
    predict_all(weights, input, 'my_answer.labeled')

#uni
'''
Accuracy = 92.702798%
'''