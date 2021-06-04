from collections import defaultdict
import dill
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class SupprtVectorMachine():
    def __init__(self):
        self.weight = defaultdict(lambda: 0)
        self.c = 0.0001
        self.margin = 0.0005
        self.lr = 1

    def load_file(self, file, labeled=True):
        X, y = [], []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if labeled:
                    label, sentence = line.split("\t")
                    label = int(label)
                else:
                    sentence = line
                    label = None
                words = sentence.split()
                X.append(words)
                y.append(label)
        return X, y

    def create_features(self, X):
        features_dicts = []
        for words in X:
            features = defaultdict(lambda: 0)
            for word in words:
                features[word] += 1
            features_dicts.append(features)
        return features_dicts
    
    def predict_one(self, features):
        y_pred = []
        for i in range(len(features)):
            score = 0
            for word, val in features[i].items():
                if word in self.weight.keys():
                    score += val * self.weight[word]
            y_pred.append(1 if score>=0 else -1)
        return y_pred

    def predict_all(self, X):
        feats = self.create_features(X)
        y_pred = self.predict_one(feats)
        return y_pred

    def normalizel1(self, sentence):
        for word in sentence:
            if word not in self.weight.keys():
                continue
            weight = self.weight[word]
            if abs(weight) < self.c:
                c = 0
            else:
                c = self.sign(weight) * self.c
            self.weight[word] -= c
        return self
    
    def sign(self, w):
        if w > 0:
            return 1
        elif w == 0:
            return 0
        else:
            return -1

    def update_weights(self, features, y):
        for word, val in features.items():
            if word in self.weight.keys():
                self.weight[word] += val * y * self.lr
        return self
            

    def train(self, X, y, iter):
        for i in range(iter):
            print("iter: ", i)
            feats = self.create_features(X)
            y_pred = self.predict_all(X)
            for idx in range(len(y)):
                self.normalizel1(X[idx])
                if y_pred[idx] * y[idx] <= self.margin:
                    self.update_weights(feats[idx], y[idx])
        return self
    
    def save_model(self, pth):
        with open(pth, 'wb') as f:
            dill.dump(self.weight, f)

    def load_model(self, pth):
        with open(pth, "rb") as f:
            self.W = dill.load(f)


if __name__ == "__main__":
    train_file = "../data/titles-en-train.labeled"
    test_file = "../data/titles-en-test.labeled"
    save_file = "model/tutorial06.dill"

    model = SupprtVectorMachine()
    train_X, train_y = model.load_file(train_file)
    test_X, test_y = model.load_file(test_file)

    model.train(train_X, train_y, iter=10)
    model.save_model(save_file)
    model.load_model(save_file)

    pred_y = model.predict_all(train_X)
    pred_test_y = model.predict_all(test_X)
    print(accuracy_score(test_y, pred_test_y))
    print(confusion_matrix(test_y, pred_test_y))
    print(classification_report(test_y, pred_test_y))