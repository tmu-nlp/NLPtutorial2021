from collections import defaultdict
from tqdm import tqdm

def make_feats(stack, queue):
    features = defaultdict(int)
    if len(stack) > 0 and len(queue) > 0:
        # (idx, word, pos)
        # (0, "ROOT", "ROOT")
        features[f'W-1_{stack[-1][1]}_W-0_{queue[0][1]}'] = 1
        features[f'W-1_{stack[-1][1]}_P-0_{queue[0][2]}'] = 1
        features[f'P-1_{stack[-1][2]}_W-0_{queue[0][1]}'] = 1
        features[f'P-1_{stack[-1][2]}_P-0_{queue[0][2]}'] = 1
    if len(stack) > 1:
        features[f'W-2_{stack[-2][1]}_W-1_{stack[-1][1]}'] = 1
        features[f'W-2_{stack[-2][1]}_P-1_{stack[-1][2]}'] = 1
        features[f'P-2_{stack[-2][2]}_W-1_{stack[-1][1]}'] = 1
        features[f'P-2_{stack[-2][2]}_P-1_{stack[-1][2]}'] = 1
    print(features)
    return features

def calc_score(w, feats, queue):
    s_shift = s_left = s_right = 0
    for key, val in feats.items():
        s_shift += w["SHIFT"][key] * val
        s_left += w["LEFT"][key] * val
        s_right += w["RIGHT"][key] * val
        if s_shift >= s_left and s_shift >= s_right and queue:
            return "SHIFT"
        elif s_left > s_right:
            return "REDUCE_LEFT"
        else:
            return "REDUCE_RIGHT"

def shift_reduce(queue, head):
    stack = [(0, "ROOT", "ROOT")]
    while len(queue) > 0 or len(stack) > 1:
        feats = make_feats(stack, queue)
        command = calc_score(w, feats, queue)
        if command == "SHIFT":
            stack.append(queue.pop(0))
        elif command == "REDUCE_LEFT":
            head[stack[-2][0]] = stack[-1][0]
            stack.pop(-2)
        else:
            head[stack[-1][0]] = stack[-2][0]
            stack.pop(-1)
        return

def train(queues, heads, w, iter=1):
    for i in tqdm(range(iter)):
        for queue, head in zip(queues, heads):
            shift_reduce(queue, head)

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

    queues, heads = load_data(train_file)
    # print(len(queues), len(heads))
    w = {
        "SHIFT": defaultdict(int),
        "LEFT": defaultdict(int),
        "RIGHT": defaultdict(int)}
    train(queues, heads, w)

    