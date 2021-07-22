from collections import defaultdict
import dill
from tqdm import tqdm


def create_features(x, y):
    phi = defaultdict(lambda: 0)
    for i in range(len(y)+1):
        first_tag = "<s>" if i == 0 else y[i-1]
        next_tag = "</s>" if i == len(y) else y[i]
        phi[create_trans(first_tag, next_tag)] += 1

    for i in range(len(y)):
        phi[create_emit(y[i], x[i])] += 1

    return phi


def create_trans(first_tag: str, next_tag: str) -> str:
    return f'T {first_tag} {next_tag}'


def create_emit(tag: str, word: str) -> str:
    return f'E {tag} {word}'


def hmm_viterbi(x: list, possible_tags: dict, trainsition: dict, w: dict, b:int):
    l = len(x)
    best_score = {}
    best_edge = {}
    best_score["0 <s>"] = 0
    best_edge["0 <s>"] = None
    active_tags = [["<s>"]]

    for i in range(l):
        my_best = {}
        for prev in active_tags[i]:
            for next in possible_tags.keys():
                if f"{i} {prev}" in best_score and f"{prev} {next}" in trainsition:
                    score = best_score[f"{i} {prev}"] + w[create_trans(prev, next)] + w[create_emit(next, x[i])]
                    if f'{i+1} {next}' not in best_score or best_score[f'{i+1} {next}'] < score:
                        best_score[f'{i+1} {next}'] = score
                        best_edge[f'{i+1} {next}'] = f'{i} {prev}'
                        my_best[next] = score
        best_b = sorted(best_score.items(), key=lambda x: x[1], reverse=True)[:b]
        active_tags.append([key.split(' ')[1] for key, value in best_b])

    for prev in active_tags[-1]:
        if f"{l} {prev}" in best_score and f"{prev} </s>" in trainsition:
            score = best_score[f"{l} {prev}"] + w[create_trans(prev, '</s>')]
            if f'{l+1} </s>' not in best_score or best_score[f'{l+1} </s>'] < score:
                best_score[f'{l+1} </s>'] = score
                best_edge[f'{l+1} </s>'] = f'{l} {prev}'

    back_step_tags = []
    next_edge = best_edge[f'{l+1} </s>']
    while next_edge != '0 <s>':
        id, tag = next_edge.split(' ')
        back_step_tags.append(tag)
        next_edge = best_edge[next_edge]
    back_step_tags.reverse()

    return back_step_tags


def test(filename: str, b: int):
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
            y_hat = hmm_viterbi(x, possible_tags, trainsition, w, b)
            print(' '.join(y_hat), file=out_f)


def main():
    test(r'C:\Users\Lexus\Documents\GitHub\Nlptutorial\data\wiki-en-test.norm', 5)


if __name__ == "__main__":
    main()