from collections import defaultdict


def predict_one(w, phi):
    score = 0
    for k,v in phi.items():
        if k in w:
            score += v * w[k]
    if score >= 0:
        return 1
    else:
        return -1

def create_features(x):
    phi = defaultdict(int)
    for word in x:
        phi['UNI:' + word] += 1
    return phi


def update_weight(w, phi, y):
    for k, v in phi.items():
        w[k] += v * y

def predict_all(model_file, input_file):
    weights = defaultdict(int)

    with open(model_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            #print(line)
            feature = line[0]
            weight = line[1]
            weights[feature] = int(weight)
            #print(weights.items())
    with open(input_file, 'r', encoding= 'utf-8') as filein, \
        open('./my_answer.txt', 'w', encoding= ' utf-8') as fileout:
        for line in filein.readlines():
            #print(line.strip().split())
            phi = create_features(line.strip().split())
            y_pred = predict_one(weights, phi)
            fileout.write(str(y_pred) + '\t' + line)


class MyPerceptron:

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
                    y_predict = predict_one(weight, phi)

                    if y_predict != label:
                        update_weight(weight, phi, label)

        with open(output_file, 'w', encoding= 'utf-8') as answer:
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





