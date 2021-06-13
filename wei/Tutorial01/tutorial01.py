from collections import defaultdict
import math


def train_unigram(train_input, train_output):
    # create a map counts and a variable total_count=0(reference to courseware)
    # counts->dict, counts words; total_count->int,total words appeared in given text
    counts = defaultdict(lambda :0)
    total_count = 0
    with open(train_input, 'r',encoding='utf-8') as train_infile, \
            open(train_output, 'w', encoding='utf-8') as train_outfile:
        lines = train_infile.readlines()
        for line in lines:
            words = line.strip().split()      # delete '\n',and make it to a words list
            words.append('</s>')
            for word in words:
                counts[word.lower()] += 1
                total_count += 1
        # print(counts)                            ->  defautdcit,{'word':int(counts),...}
        # print(len(counts), total_count)          ->  4701 35842
        for word in counts:
            probability = counts[word] / total_count
            train_outfile.write(word + '\t' + str(probability) +'\n')

def test_unigram(train_output, test_in, lambda_1, lambda_unk):
    # set some params, and create a map probabilities
    V = 1000000
    W = 0
    H = 0
    unk_words = 0
    probabilities = {}
    with open(train_out, 'r', encoding='utf-8') as modelfile, open(test_in, 'r', encoding='utf-8') as testfile:
        for line in modelfile.readlines():
            word, prob = line.split('\t')
            probabilities[word] = float(prob)
        # print(probabilities)       -> {'word': prob,...},len() is 4701
        # print(len(probabilities))
        for line in testfile.readlines():
            words = line.split()
            words.append('</s>')
            for word in words:
                word = word.lower()
                W += 1                # counts all words in file
                P = lambda_unk / V    # calculate (1−λ_1)/N
                if word in probabilities:
                    # calculate P(w_i)=λ_1 * P_(ML)(w_i) + (1−λ_1)/N
                    P += lambda_1 * probabilities[word]
                else:
                    unk_words += 1
                H += math.log(1 / P, 2)   # H = -(log2(P))
    # coverage = known words divided by all words
    return 'entropy = ' + str(H / W) + '\n' + 'coverage = ' + str((W - unk_words) / W)



if __name__ == '__main__':
    train_in = '../data/wiki-en-train.word'
    train_out = './model_out.txt'
    test_in = '../data/wiki-en-test.word'

    train_unigram(train_in, train_out)
    lambda_1 = 0.95
    lmabda_unk = 1 - lambda_1
    testout = test_unigram(train_out, test_in, 0.95, 0.01)
    print(testout)

'''
entropy = 10.487283059254489
coverage = 0.9043092522179975
    '''