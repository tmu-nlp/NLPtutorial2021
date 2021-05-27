from collections import defaultdict

def predict_one(w, phi):
    score = 0
    for key, value in phi.items():
        if key in w:
            score += value * w[key]
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
    for key, value in phi.items():
        w[key] += value * y

def predict_all(model_file, input_file):
    weights = defaultdict(int)

    with open(model_file, 'r') as f:
        model = f.readlines()

    for line in model:
        line = line.strip().split()
        feature = line[0]
        weight = line[1]
        weights[feature] = int(weight)

    with open(input_file, 'r') as f1:
        inputFile = f1.readlines()

    output = open('/users/kcnco/github/NLPtutorial2021/pan/tutorial05/answer.txt', 'w')

    for line in inputFile:
        phi = create_features(line.strip().split())
        y_pred = predict_one(weights, phi)
        output.write(str(y_pred) + "\t" + line)

    output.close()


class perceptron:
    def train_perceptron(self, model_file, output_file):
        with open(model_file) as f:
            model = f.readlines()

        weight = defaultdict(int)

        for iter in range(10):
            for line in model:
                line = line.strip().split()
                features = line[1:]
                label = int(line[0])
                phi = create_features(features)
                y_predict = predict_one(weight, phi)

                if y_predict != label:
                    update_weight(weight, phi, label)

        ans = open(output_file, 'w')
        for key, value in weight.items():
            ans.write(key + "\t" + str(value) + '\n')
        ans.close()

    def test_perceptron(self, model_file, test_file):
        predict_all(model_file, test_file)


Perceptron = perceptron()

print(Perceptron.train_perceptron('/users/kcnco/github/NLPtutorial2021/pan/tutorial05/titles-en-train.labeled','/users/kcnco/github/NLPtutorial2021/pan/tutorial05/myanswer.txt'))
print(Perceptron.test_perceptron('/users/kcnco/github/NLPtutorial2021/pan/tutorial05/myanswer.txt', '/users/kcnco/github/NLPtutorial2021/pan/tutorial05/titles-en-test.word'))

# Accuracy = 90.967056% (epoch = 1)
# Accuracy = 93.446688% (epoch = 10)
# Accuracy = 93.552958% (epoch = 100)
