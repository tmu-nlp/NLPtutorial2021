from math import log
from collections import defaultdict

def train_hmm(data_path, model_path, L = 0.95, N = 1000000):
    transition = defaultdict(lambda: 0) # 遷移
    context    = defaultdict(lambda: 0) # 文脈
    emit       = defaultdict(lambda: 0) # 生成

    with open(data_path) as f:
        for line in f.readlines():
            prev = "<s>"
            context[prev] += 1
            for wordtag in line.split():
                word, tag = wordtag.split("_")
                transition[f"{prev} {tag}"] += 1
                context[tag] += 1
                emit[f"{tag} {word}"] += 1
                prev = tag
            transition[f"{prev} </s>"] += 1

    with open(model_path, 'w') as f:
        for key, val in transition.items():
            prev, tag = key.split()
            f.write(f"T {key} {float(val)/context[prev]}\n")
        for key, val in emit.items():
            tag, word = key.split()
            f.write(f"E {key} {float(val)/context[tag]}\n")


def load_model(model_path):
    transition = {}
    emission = defaultdict(lambda: 0)
    possible_tags = {}
    with open(model_path) as f:
        for line in f.readlines():
            t, ctx, word, prob = line.split()
            possible_tags[ctx] = 1
            if t == "T":
                transition[f"{ctx} {word}"] = float(prob)
            else:
                emission[f"{ctx} {word}"] = float(prob)
    return (transition, emission, possible_tags)


def test_hmm(input, result, transition, emission, possible_tags, L=0.95, N=1000000):
    def forward(words):
        best_score = {}
        best_edge  = {}
        best_score["0 <s>"] = 0
        best_edge["0 <s>"]  = None
        for i in range(len(words)):
            for prev in possible_tags:
                for next in possible_tags:
                    if f"{i} {prev}" not in best_score:
                        continue
                    if f"{prev} {next}" not in transition:
                        continue
                    score = best_score[f"{i} {prev}"] \
                            - log(transition[f"{prev} {next}"]) \
                            - log(L * emission[f"{next} {words[i]}"] + (1-L)/N)
                    if f"{i+1} {next}" not in best_score \
                        or best_score[f"{i+1} {next}"] > score:
                        best_score[f"{i+1} {next}"] = score
                        best_edge[f"{i+1} {next}"] = f"{i} {prev}"
        for prev in possible_tags:
            if f"{len(words)} {prev}" not in best_score:
                continue
            if f"{prev} </s>" not in transition:
                continue
            score = best_score[f"{len(words)} {prev}"] \
                    - log(transition[f"{prev} </s>"])
            if f"{len(words)+1} </s>" not in best_score \
                or best_score[f"{len(words)+1} </s>"] > score:
                best_score[f"{len(words)+1} </s>"] = score
                best_edge[f"{len(words)+1} </s>"] = f"{len(words)} {prev}"
        return best_edge

    def backword(words, best_edge, file):
        tags  = []
        next_edge = best_edge[f"{len(words)+1} </s>"]
        while next_edge != f"0 <s>":
            pos, tag = next_edge.split(" ")
            tags.append(tag)
            next_edge = best_edge[next_edge]
        tags.reverse()
        file.write(' '.join(tags) + '\n')

    with open(input) as f1, open(result, 'w') as f2:
        for line in f1.readlines():
            words = line.split()
            backword(words, forward(words), f2)


if __name__ == '__main__':
    test_train  = 'test/05-train-input.txt'
    test_input  = 'test/05-test-input.txt'
    test_model  = 'test/model.txt'
    test_result = 'test/result.txt'
    train_hmm(test_train, test_model)
    test_hmm(test_input, test_result, *load_model(test_model))

    data_train = 'data/wiki-en-train.norm_pos'
    data_input = 'data/wiki-en-test.norm'
    data_model = 'data/model.txt'
    data_result = 'data/result.pos'
    train_hmm(data_train, data_model)
    test_hmm(data_input, data_result, *load_model(data_model))
    # t.pos result.pos
    # Accuracy: 90.82% (4144/4563)
    #
    # Most common mistakes:
    # NNS --> NN      45
    # NN --> JJ       27
    # JJ --> DT       22
    # NNP --> NN      22
    # VBN --> NN      12
    # JJ --> NN       12
    # NN --> IN       11
    # NN --> DT       10
    # NNP --> JJ      8
    # VBN --> JJ      7
