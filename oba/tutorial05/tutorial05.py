from collections import defaultdict
# lambdaを使うとpickleが使えないらしい
import dill
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

class Perceptron():
    def __init__(self) -> None:
        self.weight = defaultdict(lambda: 0)

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
        # vocab?
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
                if word in self.weight:
                    score += val * self.weight[word]
            y_pred.append(1 if score>=0 else -1)
        return y_pred

    def predict_all(self, X):
        feats = self.create_features(X)
        y_pred = self.predict_one(feats)
        return y_pred
        

    def update_weights(self, features, y):
        for word, val in features.items():
            self.weight[word] += val * y

    def train(self, X, y, iter=10):
        for i in range(iter):
            feats = self.create_features(X)
            y_pred = self.predict_all(X)
            for idx in range(len(y)):
                if y[idx] != y_pred[idx]:
                    self.update_weights(feats[idx], y[idx])
        print(self.weight)

    def save_model(self, pth):
        with open(pth, 'wb') as f:
            dill.dump(self.weight, f)

    def load_model(self, pth):
        with open(pth, "rb") as f:
            self.W = dill.load(f)

if __name__ == "__main__":
    train_file = "../data/titles-en-train.labeled"
    test_file = "../data/titles-en-test.labeled"
    save_file = "model/tutorial05.dill"

    model = Perceptron()
    train_X, train_y = model.load_file(train_file)
    test_X, test_y = model.load_file(test_file)

    model.train(train_X, train_y, iter=30)
    model.save_model(save_file)
    model.load_model(save_file)

    pred_y = model.predict_all(train_X)
    pred_test_y = model.predict_all(test_X)
    print(accuracy_score(test_y, pred_test_y))
    print(confusion_matrix(test_y, pred_test_y))
    print(classification_report(test_y, pred_test_y))

"""
              precision    recall  f1-score   support

          -1       0.90      0.88      0.89      1477
           1       0.87      0.89      0.88      1346

    accuracy                           0.88      2823
   macro avg       0.88      0.88      0.88      2823
weighted avg       0.88      0.88      0.88      2823
"""