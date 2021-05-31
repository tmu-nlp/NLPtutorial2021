import numpy as np
from typing import List, Dict
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict


class MyPerceptron():
    def __init__(self):
        self.w = defaultdict(lambda: 0)

    def create_features(self, sentence:str) -> Dict[str, int]:
        feature = defaultdict(lambda: 0)
        words = sentence.split()
        for word in words:
            feature[f'UNI:{word}'] += 1
        return feature

    def predict_one(self, feature:Dict[str, int]) -> int:
        score = 0
        for word, v in feature.items():
            if word in self.w:
                score += v * self.w[word]
        if score >= 0:
            return 1
        else:
            return -1

    def predict_all(self, input_file:str) -> List[int]:
        result = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                feature = self.create_features(line)
                y = self.predict_one(feature)
                result.append(y)
        return result

    def update_w(self, feature:Dict[str,int], sign:int):
        for word, cnt in feature.items():
            self.w[word] += sign * cnt

    # def check_score(self, gold_file:str, pred: List[int], detail: bool= False):
    #     gold = []
    #     with open(gold_file, 'r', encoding= 'utf-8') as f:
    #         for line in f:
    #             label = int(line.split('\t')[0])
    #             gold.append(label)
    #         gold = np.array(gold)
    #         pred = np.array(pred)
    #         if detail:
    #             print(classification_report(gold, pred))
    #         print(f'accuracy:{accuracy_score(gold, pred)}')


    def train(self, num_iterations: int, input_file:str, val_file: str= None, val_ans_file: str= None):
        with open(input_file, 'r', encoding='utf=8') as f:
            for i in range(num_iterations):
                preds = self.predict_all(val_file)
                # check_score(val_ans_file, preds)
                for line in f:
                    y, x = line.split('\t')
                    y = int(y)
                    feature = self.create_features(x)
                    y_pred = self.predict_one(feature)
                    if y != y_pred:
                        self.update_w(feature, y)


if __name__ == '__main__':
    perceptron = MyPerceptron()
    perceptron.train(10, '/Users/houjing.wei/Downloads/nlptutorial-master/data/titles-en-train.labeled',
                     '/Users/houjing.wei/Downloads/nlptutorial-master/data/titles-en-test.word',
                     '/Users/houjing.wei/Downloads/nlptutorial-master/data/titles-en-test.labeld')
    results = perceptron.predict_all('/Users/houjing.wei/Downloads/nlptutorial-master/data/titles-en-test.word')
    print('\n'.join(list(map(str, results))))
    # check_score('/Users/houjing.wei/Downloads/nlptutorial-master/data/titles-en-test.labeled', results, True)




