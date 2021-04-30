from collections import defaultdict
from math import log2


def train_unigram(file):
    counts = defaultdict(lambda: 0)
    total_count = 0
    model = {}
    with open(file) as f:
        for line in f.readlines():
            words = line.split() + [ '</s>' ]
            for word in words:
                counts[word] += 1
                total_count  += 1
    for word in counts:
        prob = float(counts[word]) / total_count
        model[word] = prob
    lmodel = ['{} {}\n'.format(str(w), str(p)) for w, p in model.items()]
    with open('model_file.txt', mode='w') as f:
        f.writelines(lmodel)
    return model


def test_unigram(model_file, test_file, lambda_1=0.95, lambda_unk=0.05, V=1000000):
    def load_model(model_file):
        prob = {}
        with open(model_file) as f:
            for line in f.readlines():
                w, p = line.split()
                prob[w] = p
        return prob

    def evaluate(test_file, prob, lambda_1, lambda_unk, V):
        W=0
        H=0
        unk = 0
        with open(test_file) as f:
            for line in f.readlines():
                words = line.split() + [ '</s>' ]
                for w in words:
                    W += 1
                    P = lambda_unk / V
                    if w in prob:
                        P += lambda_1 * float(prob[w])
                    else:
                        unk += 1
                    H += -1 * log2(P)
        print(f"entropy = {H / W}")
        print(f"coverage = {(W-unk) / W}")

    evaluate(test_file, load_model(model_file), lambda_1, lambda_unk, V)


if __name__ == '__main__':
    train_unigram('data/wiki-en-train.word')
    test_unigram('model_file.txt', 'data/wiki-en-test.word')
