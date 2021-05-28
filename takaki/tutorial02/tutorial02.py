from collections import defaultdict
from math import log2


class Tutorial02:
    model = defaultdict(float)

    def train(self, path):
        counts  = defaultdict(int)
        context = defaultdict(int)
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            words = ['<s>'] + line.split() + ['</s>']
            for i in range(1, len(words)):
                counts[f'{words[i-1]} {words[i]}'] += 1
                context[words[i-1]] += 1
                counts[words[i]]    += 1
                context['']         += 1
        for ngram, count in counts.items():
            words = ngram.split()
            word = '' if len(words) < 2 else words[0]
            self.model[ngram] = float(count) / context[word]

    def test_interp(self, path, λ1=0.5, λ2=0.5, V=1000000):
        W, H = 0, 0
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            words = ['<s>'] + line.split() + ['</s>']
            for i in range(1, len(words)):
                P1 = λ1 * self.model[words[i]] + float(1 - λ1) / V
                P2 = λ2 * self.model[f'{words[i-1]} {words[i]}'] + (1 - λ2) * P1
                H -= log2(P2)
                W += 1
        return H / W

    def test_wittenbell(self, path, λ1=0.5, V=1000000):
        W, H = 0, 0
        memo = defaultdict(lambda: defaultdict(int))
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            words = ['<s>'] + line.split() + ['</s>']
            for i in range(1, len(words)):
                memo[words[i-1]][words[i]] += 1
        for line in lines:
            words = ['<s>'] + line.split() + ['</s>']
            for i in range(1, len(words)):
                U = len(memo[words[i-1]])
                C = sum(memo[words[i-1]].values())
                λ2 = 1 - (float(U) / (U + C))
                P1 = λ1 * self.model[words[i]] + float(1 - λ1) / V
                P2 = λ2 * self.model[f'{words[i-1]} {words[i]}'] + (1 - λ2) * P1
                H -= log2(P2)
                W += 1
        return H / W

    def store(self, path):
        with open(path, 'w') as f:
            f.writelines([f'{n}\t{p}\n' for n, p in self.model.items()])

    def restore(self, path):
        with open(path, 'r') as f:
            self.model = {
                x[0]: float(x[1]) for l in f.readlines() if (x := l.split())
            }


if __name__ == '__main__':
    x = Tutorial02()
    x.train('data/02-train-input.txt')
    x.store('data/model.txt')

    x = Tutorial02()
    x.train('data/wiki-en-train.word')
    entropy = x.test_interp('data/wiki-en-test.word', λ1=0.8, λ2=0.4)
    print(f'entropy: {entropy}')
    '''
    entropy: 9.666736125857843
    '''

    entropy = min({
        λ1: x.test_wittenbell('data/wiki-en-test.word', λ1=λ1)
        for i in range(100)
        if (λ1 := i/100)
    }.items(), key=lambda x: x[1])
    print(f'entropy: {entropy[1]} (λ1 = {entropy[0]})')
    '''
    entropy: 9.6850028730648 (λ1 = 0.82)
    '''
