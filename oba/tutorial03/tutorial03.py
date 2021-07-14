import sys
import math
sys.path.append("../")
from tutorial01.tutorial01 import UnigramLangModel

class Tokenizer(UnigramLangModel):
    def __init__(self):
        super().__init__()
        self.word_probs = dict()

    def load_sentences(self, test_path):
        with open(test_path, "r") as f:
            sentences = f.readlines()
        return [sentence.strip() for sentence in sentences]

    def forward(self, sentence):
        V = 1e6
        best_edge = [None] * (len(sentence)+1)
        best_score = [float("inf")] * (len(sentence)+1)
        best_score[0] = 0
        for word_end in range(1, len(sentence)+1): # 01→02→01→03→13→23...
            for word_begin in range(word_end):
                word = sentence[word_begin:word_end]
                if word in self.word_probs or len(word) == 1:
                    prob = (1 - self.λ_1)/V
                    if word in self.word_probs: # 未知語でない場合
                        prob += self.λ_1 * self.word_probs[word]
                    my_score = best_score[word_begin] + (-1) * math.log2(prob)
                    if my_score < best_score[word_end]:
                        best_score[word_end] = my_score
                        best_edge[word_end] = (word_begin, word_end)
        return best_edge
    
    def backward(self, sentence, best_edge):
        words = []
        next_edge = best_edge[-1]
        while next_edge != None:
            words.append(sentence[next_edge[0]:next_edge[1]])
            next_edge = best_edge[next_edge[0]]
        return " ".join(reversed(words))

    def tokenize(self, test_file):
        results = []
        sentences = self.load_sentences(test_file)
        for sentence in sentences:
            best_edge = self.forward(sentence)
            result = self.backward(sentence, best_edge)
            results.append(result)
        return results
    
if __name__ == "__main__":
    tokenizer = Tokenizer()

    # train
    train_file = "../data/wiki-ja-train.word"
    model_file = "tutorial03.txt"
    tokenizer.train(train_file_pth=train_file).save(prob_file=model_file)

    # test
    test_file = "../data/wiki-ja-test.txt"
    tokenized_file = "tutorial03_tokenized.txt"
    tokenized = tokenizer.tokenize(test_file)
    with open(tokenized_file, "w") as f:
        for token in tokenized:
            f.write(token+"\n")

"""
Sent Accuracy: 23.81% (20/84)
Word Prec: 68.93% (1861/2700)
Word Rec: 80.77% (1861/2304)
F-meas: 74.38%
Bound Accuracy: 83.25% (2683/3223)
"""