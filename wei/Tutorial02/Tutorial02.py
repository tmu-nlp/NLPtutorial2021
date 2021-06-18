from collections import Counter
from collections import defaultdict
import math
import string


class Ngram():
    def __init__(self, lambda_x):
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)       # create counts,context_counts
        self.lambada_1 = 0.95
        self.lambada_2 = lambda_x / 100
        self.V = 1000000
        self.W = 0
        self.H = 0

    def train(self,inputFile):
        with open(inputFile, encoding='gb18030', errors='ignore') as file:
            for line in file:
                words = line.translate(str.maketrans('', '', string.punctuation))
                words = words.lower().strip().split()
                words.insert(0, "<s>")
                words.append('</s>')
                for i in range(1,len(words)-1):                       # 1~len-1
                    ngramwords =' '.join([words[i-1], words[i]])
                    self.ngram_counts[ngramwords] += 1
                    print(self.ngram_counts)
                    self.context_counts[words[i]] += 1

        # output_File = open(outputFile,'w',encoding='gb18030', errors='ignore')
        for ngram, count in self.ngram_counts.items():
            bigram = ngram.split()                                       # 'w_(i-1)w_1' --> 'w_(i-1)',w_1'
            bigram = ' '.join(bigram[:-1])
            probability = count / self.context_counts[bigram]
            model[ngram] = probability
        return model

    def test(self,model_file,testfile):
        with open(testfile,'r',encoding='gb18030', errors='ignore') as file:
            for line in file:
                words = line.lower().strip().split()
                words.insert(0, '<s>')
                words.append('</s>')

                for i in range(1,len(words)-1):
                    P1 = self.lambada_1 * model_file[words[i]] + (1 - self.lambada_1)/self.V
                    P2 = self.lambada_2 * model_file[''.join([words[i-1],words[i]])]+(1-self.lambada_2)*P1
                    self.H +=  math.log(1/(P2, 2))
                    self.W += 1
                return print(f'lambda_2 = {self.lambada_2}: entropy = {self.H / self.W}')






if __name__ == '__main__':
    trainpath = '../data/wiki-en-train.word'
    testpath = '../data/wiki-en-test.word'

    for i in range(5, 100, 5):
        bigramLM = Ngram(i)
        model_file = bigramLM.train(trainpath)
        bigramLM.test(model_file, testpath)
