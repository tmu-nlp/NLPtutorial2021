from collections import defaultdict
import math

def train_hmm(train_file, model_file):
    with open(train_file) as file, open(model_file, 'w') as m_file:
        emit = defaultdict(lambda: 0) #品詞→単語
        transition = defaultdict(lambda: 0) #品詞→品詞
        context = defaultdict(lambda: 0) #文脈の頻度

        for line in file:
            previous = '<s>'
            context[previous] += 1
            wordtags = line.split()
            for wordtag in wordtags:
                word, tag = wordtag.split('_')
                transition[previous+' '+tag] += 1
                context[tag] += 1
                emit[tag+' '+word] += 1
                previous = tag
            transition[previous+' </s>'] += 1

        #遷移確率
        for key, value in transition.items():
            previous, word = key.split()
            #print('Transition', key, value/context[previous])
            m_file.write(f"T {key} {value/context[previous]}\n")

        #生成確率
        for key, value in emit.items():
            previous, word = key.split()
            #print('Emit', key, value/context[previous])
            m_file.write(f"E {key} {value/context[previous]}\n")


def test_hmm(model_file, test_file, output_file):
    #モデル読み込み
    transition = defaultdict(lambda: 0)
    emission = defaultdict(lambda: 0)
    possible_tags = {}

    with open(model_file) as m_file:
        for line in m_file:
            type, context, word, prob = line.split()
            possible_tags[context] = 1
            if type == 'T':
                transition[f'{context} {word}'] = float(prob)
            else:
                emission[f'{context} {word}'] = float(prob)

    #viterbiアルゴリズム
    with open(test_file) as file, open(output_file, 'w') as o_file:
        for line in file:
            #forward step
            lambda_1 = 0.95; N = 10**6
            words = line.split()
            words.append('</s>')
            length = len(words)
            best_score, best_edge = {}, {}
            best_score['0 <s>'] = 0
            best_edge['0 <s>'] = None
            for i in range(length):
                for prev in possible_tags.keys():
                    for next in possible_tags.keys():
                        if f'{i} {prev}' in best_score and f'{prev} {next}' in transition:
                            score = best_score[f'{i} {prev}'] \
                                    - math.log(transition[f'{prev} {next}'], 2) \
                                    - math.log(lambda_1 * emission[f'{next} {words[i]}'] + (1-lambda_1) / N, 2)
                            if f'{i+1} {next}' not in best_score or best_score[f'{i+1} {next}'] > score:
                                best_score[f'{i+1} {next}'] = score
                                best_edge[f'{i+1} {next}'] = f'{i} {prev}'
            #</s>に対する操作     
            best_score[f'{length+1} </s>'] = float('inf')
            for prev in possible_tags.keys():
                    if f'{length} {prev}' in best_score and f'{prev} </s>' in transition:
                        score = best_score[f'{length} {prev}'] \
                                - math.log(transition[f'{prev} </s>'], 2)
                        if f'{length+1} {next}' not in best_score or best_score[f'{length+1} </s>'] > score:
                            best_score[f'{length+1} </s>'] = score
                            best_edge[f'{length+1} </s>'] = f'{length} {prev}'
            #backward step
            tags = []
            next_edge = best_edge[f'{length+1} </s>']
            while next_edge != '0 <s>':
                position, tag = next_edge.split()
                tags.append(tag)
                next_edge = best_edge[next_edge]
            tags.reverse()
            #print(' '.join(tags[:-1])) ?
            o_file.write(' '.join(tags[:-1])+'\n')


if __name__ == '__main__':
    train_file = '../data/wiki-en-train.norm_pos'
    model_file = 'tutorial04_model.txt'
    train_hmm(train_file, model_file)

    test_file = '../data/wiki-en-test.norm'
    output_file = 'my_answer.pos'
    test_hmm(model_file, test_file, output_file)


"""
Accuracy: 90.75% (4141/4563)

Most common mistakes:
NNS --> NN      45
NN --> JJ       27
JJ --> DT       22
NNP --> NN      22
JJ --> NN       12
VBN --> NN      12
NN --> IN       11
NN --> DT       10
NNP --> JJ      8
VBP --> VB      7
"""