from collections import defaultdict
import dill
from tqdm import tqdm

def create_trans(first_tag: str, next_tag: str) -> str:
    return f'T {first_tag} {next_tag}'


def create_emit(tag: str, word: str) -> str:
    return f'E {tag} {word}'

def create_features(x, y):
    phi = defaultdict(lambda: 0)
    for i in range(len(y)+1):
        first_tag = "<s>" if i == 0 else y[i-1]
        # if i == 0:
        #     first_tag = "<s>"
        # else:
        #     first_tag = y[i-1]

        next_tag = "</s>" if i == len(y) else y[i]
        # if i == len(y):
        #     next_tag = "</s>"
        # else:
        #     next_tag = y[i]

        phi[create_trans(first_tag, next_tag)] += 1

    for i in range(len(y)):
        phi[create_emit(y[i], x[i])] += 1

    return phi



def hmm_viterbi(x: list, possible_tags: dict, trainsition: dict, w: dict):
    l = len(x)
    best_score = defaultdict(lambda: 0)
    best_edge = defaultdict(lambda: 0)
    best_score["0 <s>"] = 0
    best_edge["0 <s>"] = None
    for i in range(l):
        for prev in possible_tags.keys():
            for next in possible_tags.keys():
                if f"{i} {prev}" in best_score and f"{prev} {next}" in trainsition:
                    score = best_score[f"{i} {prev}"] + w[create_trans(prev, next)] + w[create_emit(next, x[i])]
                    if f'{i+1} {next}' not in best_score or best_score[f'{i+1} {next}'] < score:
                        best_score[f'{i+1} {next}'] = score
                        best_edge[f'{i+1} {next}'] = f'{i} {prev}'

    #for prev in possible_tags.keys():
    #    if f"{l} {prev}" in best_score and f"{prev} </s>" in trainsition:
    #        score = best_score[f"{l} {prev}"] + w[create_trans(prev, next)]
    #        if f'{l+1} </s>' not in best_score or best_score[f'{l+1} </s>'] < score:
    #            best_score[f'{l+1} </s>'] = score
    #            best_edge[f'{l+1} </s>'] = f'{l} {prev}'

    back_step_tags = []
    next_edge = best_edge[f'{l+1} </s>']
    while next_edge != '0 <s>':
        id, tag = next_edge.split(' ')
        back_step_tags.append(tag)
        next_edge = best_edge[next_edge]
    back_step_tags.reverse()

    return back_step_tags


def load_data(filename: str, trainsition: dict, possible_tags: dict) -> dict:
    x = []
    y = []
    possible_tags['<s>'] = 1
    possible_tags['</s>'] = 1

    with open(filename, 'r') as f:
        for line in f:
            word_tags = line.strip().split(' ')
            x_line = []
            y_line = []
            prev_tag = '<s>'
            for word_tag in word_tags:
                word, tag = word_tag.split('_')
                x_line.append(word)
                y_line.append(tag)
                trainsition[f'{prev_tag} {tag}'] = 1
                prev_tag = tag
                possible_tags[tag] = 1
            trainsition[f'{prev_tag} </s>'] = 1
            x.append(x_line)
            y.append(y_line)

    return x, y


def train(filename: str):
    w = defaultdict(lambda: 0)
    trainsition = defaultdict(lambda: 0)
    possible_tags = defaultdict(lambda: 0)
    x, y = load_data(filename, trainsition, possible_tags)
    
    for _ in tqdm(range(5), desc='epoch'):
        for i in tqdm(range(len(x)), desc='train'):
            y_hat = hmm_viterbi(x[i], possible_tags, trainsition, w)
            phi_prime = create_features(x[i], y[i])
            phi_hat = create_features(x[i], y_hat)

            for key, value in phi_prime.items():
                w[key] += value
            for key, value in phi_hat.items():
                w[key] -= value

    with open('hmm_pr', 'wb') as hmm_f,\
        open('possible_t', 'wb') as p_f, \
        open('tl', 'wb') as t_f:
        hmm_f.write(dill.dumps(w))
        p_f.write(dill.dumps(possible_t))
        t_f.write(dill.dumps(tl))


def test(filename: str):
    with open('hmm_percep', 'rb') as hmm_f,\
        open('possible_tags', 'rb') as p_f, \
        open('trainsition', 'rb') as t_f:
        w = dill.loads(hmm_f.read())
        possible_tags = dill.loads(p_f.read())
        trainsition = dill.loads(t_f.read())


    with open('out.txt', 'w') as out_f,\
        open(filename, 'r') as input_f:
        for line in tqdm(input_f, desc='test'):
            x = line.strip().split(' ')
            y_hat = hmm_viterbi(x, possible_tags, trainsition, w)
            print(' '.join(y_hat), file=out_f)

def main():
    # train('../../test/05-train-input.txt')
    # test('../../test/05-test-input.txt')
    train('../../data/wiki-en-train.norm_pos')
    test('../../data/wiki-en-test.norm')


if __name__ == "__main__":
    main()