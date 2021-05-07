from collections import defaultdict
import math

class BigramLanguageModel:
    def __init__(self):
        self.cnts = defaultdict(lambda: 0)
        self.context_cnts = defaultdict(lambda: 0)
        self.probs = defaultdict(lambda: 0)
        
    def train(self, filename):
        with open(filename) as f:
            for line in f:
                words = line.split()
                words.insert(0, "<s>")
                words.append("<\s>")
                for i in range(1, len(words)):
                    self.cnts[f"{words[i-1]} {words[i]}"] += 1
                    self.context_cnts[words[i-1]] += 1
                    self.cnts[words[i]] += 1
                    self.context_cnts[""] += 1
        for ngram, cnt in self.cnts.items():
            words = ngram.split()
            if len(words) == 1:
                self.probs[ngram] = self.cnts[ngram] / self.context_cnts[""]
            else:
                self.probs[ngram] = self.cnts[ngram] / self.context_cnts[words[0]]
       
    def test(self, filename, lambda_1, lambda_2):
        V = 10**6; W = 0; H = 0
        with open(filename) as f:
            for line in f:
                words = line.split()
                words.insert(0, "<s>")
                words.append("<\s>")
                for i in range(1, len(words)):
                    p1 = lambda_1 * self.probs[words[i]] + (1 - lambda_1) / V
                    p2 = lambda_2 * self.probs[f"{words[i-1]} {words[i]}"] + (1 - lambda_2) * p1
                    H -= math.log(p2, 2)
                    W += 1
            entropy = H / W
        
        return entropy
        
    def grid_search(self, filename):
        min_entropy = float("INF")
        lambda1 = 0
        lambda2 = 0
        for lambda_1 in range(5, 100, 5):
            for lambda_2 in range(5, 100, 5):
                lambda_1 /= 100
                lambda_2 /= 100
                entropy = self.test(filename, lambda_1, lambda_2)
                if entropy < min_entropy:
                    min_entropy = entropy
                    lambda1 = lambda_1
                    lambda2 = lambda_2
        
        return min_entropy, lambda1, lambda2
        
        
if __name__ == "__main__":
    BigramLM = BigramLanguageModel()
    
    train_file = "../data/wiki-en-train.word"
    test_file = "../data/wiki-en-test.word"
    
    BigramLM.train(train_file)
    result = BigramLM.grid_search(test_file)
    print(f"Entropy: {result[0]}  (lambda1: {result[1]}, lambda2: {result[2]})")