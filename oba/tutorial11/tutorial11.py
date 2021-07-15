from collections import defaultdict
from tqdm import tqdm

def make_feats(stack, queue):
    features = defaultdict(int)
    if len(stack) > 0 and len(queue) > 0:
        # (idx, word, pos)
        # (0, "ROOT", "ROOT")
        # stackの一番後ろとqueueの一番前の素性を数える
        features[f'W-1_{stack[-1][1]}_W-0_{queue[0][1]}'] = 1
        features[f'W-1_{stack[-1][1]}_P-0_{queue[0][2]}'] = 1
        features[f'P-1_{stack[-1][2]}_W-0_{queue[0][1]}'] = 1
        features[f'P-1_{stack[-1][2]}_P-0_{queue[0][2]}'] = 1
    if len(stack) > 1:
        # stackの後ろから二番目と一番後ろの素性を数える
        features[f'W-2_{stack[-2][1]}_W-1_{stack[-1][1]}'] = 1
        features[f'W-2_{stack[-2][1]}_P-1_{stack[-1][2]}'] = 1
        features[f'P-2_{stack[-2][2]}_W-1_{stack[-1][1]}'] = 1
        features[f'P-2_{stack[-2][2]}_P-1_{stack[-1][2]}'] = 1
    return features

def calc_score(w, feats, queue):
    s_shift = s_left = s_right = 0
    for key, val in feats.items(): # key：W-2_word2_P-1_word1
        s_shift += w["SHIFT"][key] * val
        s_left += w["LEFT"][key] * val
        s_right += w["RIGHT"][key] * val
        if s_shift >= s_left and s_shift >= s_right and queue:
            return "SHIFT"
        elif s_left > s_right:
            return "REDUCE_LEFT"
        else:
            return "REDUCE_RIGHT"

def calc_corr(stack, queue, heads, unproc):
    # stack[idx][0]：idx番目の単語のid
    if heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
        corr = "REDUCE_RIGHT"
    elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
        corr = "REDUCE_LEFT"
    else:
        corr = "SHIFT"
    return corr

def update_weights(weights, feats, command, correct):
    for key, value in feats.items():
        weights[command][key] -= value
        weights[correct][key] += value

def shift_reduce(queue, w):
    stack = [(0, "ROOT", "ROOT")]
    heads = [-1] * (len(queue) + 1)
    while len(queue) > 0 or len(stack) > 1:
        feats = make_feats(stack, queue)
        command = calc_score(w, feats, queue)
        if command == "SHIFT":
            stack.append(queue.pop(0))
        elif command == "REDUCE_LEFT":
            heads[stack[-2][0]] = stack[-1][0]
            stack.pop(-2)
        else:
            heads[stack[-1][0]] = stack[-2][0]
            stack.pop(-1)
        return heads

def shift_reduce_train(queue, head, w, stack, unproc):
    # i番目の出現回数
    feats = make_feats(stack, queue)
    
    command = calc_score(feats, queue)
    if command == "SHIFT":
        stack.append(queue.pop(0))
    elif command == "REDUCE_LEFT":
        head[stack[-2][0]] = stack[-1][0]
        stack.pop(-2)
    else:
        head[stack[-1][0]] = stack[-2][0]
        stack.pop(-1)
    
    correct = calc_corr(stack, queue, heads, unproc)
    if correct == "SHIFT":
        stack.append(queue.pop(0))
    elif correct == "REDUCE_LEFT":
        unproc[stack[-2][0]] -= 1
        del stack[-1]
    else:
        unproc[stack[-1][0]] -= 1
        del stack[-2]

    if command != correct:
        update_weights(w, feats)

def train_sr(queues, heads, w, iter=1):
    for i in tqdm(range(iter)):
        for queue, head in zip(queues, heads):
            stack = [(0, "ROOT", "ROOT")]
            unproc = [heads.count(i) for i in range(heads)]
            shift_reduce_train(queue, head, w, stack, unproc)

def test_sr(test_queues, w, result_path):
    with open(result_path, 'w') as f:
        for queue in test_queues:
            heads = shift_reduce(queue, w)
            for head in heads[1:]:
                f.write("\t" * 6 + f"{head}\t_")
            f.write("\n")


def load_data(file):
        queues, queue, heads, head = [], [], [], [-1]
        with open(file, "r") as f:
            for line in f:
                line = line.strip().split()
                if len(line) != 0:
                    id, word, pos, label = line[0], line[2], line[3], line[7]
                    id = int(id)
                    queue.append((id, word, pos))
                    head.append(label)
                else:
                    queues.append(queue)
                    heads.append(head)
                    queue, head = [], [-1]
        return queues, heads

if __name__ == "__main__":
    train_file = "data/mstparser-­en-­train.dep"
    train_file = train_file.replace("\xad", "")
    test_file = "data/mstparser­-en­-test.dep"
    result_file = "tutorial11/tutorial11.txt"

    queues, heads = load_data(train_file)
    test_queues, test_heads = load_data(test_file)
    # print(len(queues), len(heads))
    w = {
        "SHIFT": defaultdict(int),
        "LEFT": defaultdict(int),
        "RIGHT": defaultdict(int)}
    train_sr(queues, heads, w)
    test_sr(test_queues, test_heads, w, result_file)