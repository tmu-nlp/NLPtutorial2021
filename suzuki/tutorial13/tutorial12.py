from collections import defaultdict
from tqdm import tqdm

def hmm_viterbi(X, possible_tags, weights, transition):
    best_score = defaultdict(lambda: 0)
    best_edge = {}
    best_score['0 <s>'] = 0
    best_edge['0 <s>'] = None
    for i in range(len(X)):
        for prev in possible_tags:
            for next in possible_tags:
                if f'{i} {prev}' in best_score and f'{prev} {next}' in transition:
                    score = best_score[f'{i} {prev}'] + weights[f'T {prev} {next}'] + weights[f'E {next} {X[i]}']
                    if f'{i+1} {next}' not in best_score or best_score[f'{i+1} {next}'] < score:
                        best_score[f'{i+1} {next}'] = score
                        best_edge[f'{i+1} {next}'] = f'{i} {prev}'
    for tag in possible_tags:
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

def create_features(X, Y):
    phi = defaultdict(lambda: 0)
    for i in range(len(Y)+1):
        if i == 0: first_tag = '<s>'
        else: first_tag = Y[i-1]
        if i == len(Y): next_tag = '</s>'
        else: next_tag = Y[i]
        phi[f'T {first_tag} {next_tag}'] += 1 # CREATE_TRANS
    for i in range(len(Y)):
        phi[f'E {Y[i]} {X[i]}'] += 1
    return phi

def update_weights(weights, phi_prime, phi_hat):
    for key, value in phi_prime.items():
        weights[key] += value
    for key, value in phi_hat.items():
        weights[key] -= value

if __name__ == '__main__':
    #train_input = 'wiki-en-train.norm_pos'
    #test_input = 'wiki-en-test.norm'
    train_input = '05-train-input.txt'
    test_input = '05-test-input.txt'
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
                Y_hat = hmm_viterbi(l, possible_tags, weights, transition)
                ans.write(' '.join(Y_hat) + '\n')

'''

Accuracy: 88.19% (4024/4563)

Most common mistakes:
NN --> NNS      55
VBN --> NNS     29
NNS --> NN      24
NN --> NNP      23
JJ --> NNP      23
JJ --> NN       21
NN --> JJ       16
NNS --> NNP     11
JJ --> NNS      11
VBP --> VB      10

'''