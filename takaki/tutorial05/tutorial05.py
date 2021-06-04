from collections import defaultdict
from pprint import pprint


class NlpTutorial5:
    w = defaultdict(float)

    def create_features(self, line):
        phi = defaultdict(int)
        for word in line.split():
            phi[f"UNI:{word}"] += 1
        return phi

    def update_weights(self, phi, y):
        for name, value in phi.items():
            self.w[name] += float(value) * y

    def predict_one(self, phi):
        score = 0.0
        for name, value in phi.items():
            if name in self.w:
                score += float(value) * self.w[name]
        return 1 if score >= 0 else -1

    def predict(self, lines):
        return [self.predict_one(self.create_features(line.strip())) for line in lines]

    def train(self, data, iter=100):
        for i in range(iter):
            for x, y in data.items():
                phi = self.create_features(x)
                yn  = self.predict_one(phi)
                if y != yn:
                    self.update_weights(phi, y)

    def load_data(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        return {
            data[1]: int(data[0])
            for line in lines
            if (data := line.strip().split('\t'))
        }

    def store(self, path):
        lines = [f'{x}\t{y}\n' for x, y in self.w.items()]
        with open(path, 'w') as f:
            f.writelines(lines)

    def restore(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        w = defaultdict(float)
        for line in lines:
            data = line.split()
            w[data[0]] = float(data[1])



if __name__ == '__main__':
    x = NlpTutorial5()
    x.train(x.load_data('data/03-train-input.txt'))
    x.store('data/model.txt')

    x = NlpTutorial5()
    x.train(x.load_data('data/titles-en-train.labeled'), iter=30)
    with open('data/titles-en-test.word', 'r') as f:
        lines = f.readlines()
    result = x.predict(lines)
    result = [f'{label}\t{line}' for label, line in zip(result, lines)]
    with open('data/result.labeled', 'w') as f:
        f.writelines(result)
    '''
    py titles-en-test.labeled result.labeled
    Accuracy = 93.446688%
    '''
