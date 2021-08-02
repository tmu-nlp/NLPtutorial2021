from collections import defaultdict
import math
import sys


def main():
    args = sys.argv
    if len(args) > 1 and args[1] == "test":
        train_path = 'test/05-train-input.txt'
        test_path = 'test/05-test-input.txt'
    else:
        train_path = 'data/wiki-en-train.norm_pos'
        test_path = 'data/wiki-en-test.norm'

    emission, transition, possible_tags = train_hmm(train_path)
    test_hmm(test_path, emission, transition, possible_tags)


def train_hmm(path):
    emit = defaultdict(lambda: 0)
    trans = defaultdict(lambda: 0)
    context = defaultdict(lambda: 0)
    possible_tags = set()
    possible_tags.add('</s>')
    for line in open(path, 'r'):
        line = line.strip('\n')
        previous_tag = '<s>'
        context[previous_tag] += 1
        word_tags = line.split(' ')
        for word_tag in word_tags:
            word_tag = word_tag.split('_')
            word = word_tag[0].lower()
            tag = word_tag[1]

            trans[f'{previous_tag}|{tag}'] += 1
            emit[f'{word}|{tag}'] += 1
            context[tag] += 1
            possible_tags.add(tag)

            previous_tag = tag

        trans[f'{previous_tag}|</s>'] += 1
        emit[f'</s>|</s>'] += 1
        context['</s>'] += 1

    transition = defaultdict(float)
    for key, transition_count in trans.items():
        previous = key.split('|')[0]
        transition[key] = transition_count / context[previous]

    emission = defaultdict(float)
    for key, emit_count in emit.items():
        previous = key.split('|')[1]
        emission[key] = emit_count / context[previous]

    return emission, transition, possible_tags


def test_hmm(path, emission, transition, possible_tags):
    with open('answer', 'w') as f:
        for line in open(path, 'r'):
            words = line.strip().split()
            pred_tags = beam_search(words, emission, transition, possible_tags)
            print(' '.join(pred_tags), file=f)


def beam_search(words, emission, transition, possible_tags):
    lam, V, beam_size = 0.95, 1000000, 10
    
    words.append('</s>')
    length = len(words)
    best_score = defaultdict(lambda: 10 ** 10)
    best_score['0|<s>'] = 0
    best_edge = {f'0|<s>': None}
    active_tags = {0: ['<s>']}
    for i in range(0, length):
        beam = {}
        for prev_tag in active_tags[i]:
            i_prev = f'{i}|{prev_tag}'
            for next_tag in possible_tags:
                prev_next = f'{prev_tag}|{next_tag}'
                word_next = f'{words[i]}|{next_tag}'
                i_1_next = f'{i+1}|{next_tag}'
                if i_prev in best_score.keys() and prev_next in transition:
                    score = best_score[i_prev]
                    score += -math.log(transition[prev_next])
                    score += - \
                        math.log((lam * emission[word_next]) + ((1 - lam) / V))
                    if best_score[i_1_next] > score:
                        best_score[i_1_next] = score
                        best_edge[i_1_next] = i_prev
                        beam[next_tag] = score

        active_tags[i + 1] = []
        for tag in sorted(beam.items(), key=lambda x: x[1]):
            active_tags[i + 1].append(tag[0])
            if len(active_tags[i + 1]) == beam_size:
                break

    pred_tags = []
    next_edge = best_edge[f'{len(words)}|</s>']
    while next_edge != '0|<s>':
        tag = next_edge.split('|')[1]
        pred_tags.append(tag)
        next_edge = best_edge[next_edge]
    pred_tags.reverse()
    return pred_tags


if __name__ == '__main__':
    main()
