from collections import defaultdict
from tqdm import tqdm
from tutorial12 import hmm_viterbi, create_features, update_weights

def test_hmm_beam(X, possible_tags, weights, transition):
    best_score = defaultdict(lambda: 0)
    best_edge = {}
    active_tags = defaultdict(list)
    best_score['0 <s>'] = 0
    best_edge['0 <s>'] = None
    active_tags[0] = ['<s>']
    for i in range(len(X)):
        my_best = defaultdict(lambda: 0)
        for prev in active_tags[i]:
            for next in possible_tags:
                if f'{i} {prev}' in best_score and f'{prev} {next}' in transition:
                    score = best_score[f'{i} {prev}'] + weights[f'T {prev} {next}'] + weights[f'E {next} {X[i]}']
                    if f'{i+1} {next}' not in best_score or best_score[f'{i+1} {next}'] < score:
                        best_score[f'{i+1} {next}'] = score
                        best_edge[f'{i+1} {next}'] = f'{i} {prev}'
                        my_best[f'{next}'] = score
        active_tags[i+1] = list(my_best.keys())
    for tag in active_tags[len(X)]:
        if f'{tag} </s>' in transition:
            score = best_score[f'{len(X)} {tag}'] + weights[f'T {tag} </s>']
            if f'{len(X)+1} </s>' not in best_score or best_score[f'{len(X)+1} </s>'] < score:
                best_score[f'{len(X)+1} </s>'] = score
                best_edge[f'{len(X)+1} </s>'] = f'{len(X)} {tag}'
    
    tags = []
    next_edge = best_edge[f'{len(X)+1} </s>']
    while next_edge != '0 <s>':
        _, tag = next_edge.split(' ')
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags = tags[::-1]

    return tags

if __name__ == '__main__':
    train_input = 'wiki-en-train.norm_pos'
    test_input = 'wiki-en-test.norm'
    #train_input = '05-train-input.txt'
    #test_input = '05-test-input.txt'
    iterations = 5

    #preprocess
    possible_tags = {'<s>', '</s>'}
    transition = set()
    data = []
    with open(train_input, 'r') as f:
        for line in f:
            sentence = []
            y_primes = []
            x_ys = line.strip().split(' ')
            prev = '<s>'
            for x_y in x_ys:
                x, y = x_y.split('_')
                sentence.append(x)
                y_primes.append(y)
                possible_tags.add(y)
                transition.add(f'{prev} {y}')
                prev = y
            transition.add(f'{prev} </s>')

            data.append((tuple(sentence), tuple(y_primes)))

    #train
    weights = defaultdict(lambda: 0)
    for i in tqdm(range(iterations)):
        for X, Y_prime in data:
            Y_hat = hmm_viterbi(X, possible_tags, weights, transition)
            phi_prime = create_features(X,Y_prime)
            phi_hat = create_features(X, Y_hat)
            update_weights(weights, phi_prime, phi_hat)
    
    #test
    with open(test_input, 'r') as f:
        with open('my_answer.pos', 'w') as ans:
            for line in f:
                l = line.strip().split(' ')
                Y_hat = test_hmm_beam(l, possible_tags, weights, transition)
                ans.write(' '.join(Y_hat) + '\n')

'''

$ perl gradepos.pl wiki-en-test.pos my_answer.pos

Accuracy: 87.68% (4001/4563)

Most common mistakes:
NN --> JJ       40
NN --> NNP      31
NN --> NNS      30
NNS --> NN      24
NN --> VBN      17
JJ --> NNP      17
NNP --> JJ      15
NN --> VBG      13
NNS --> NNP     12
VBP --> VB      12

'''