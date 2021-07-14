import subprocess
from tqdm import tqdm
from collections import defaultdict

SHIFT, LEFT, RIGHT = range(3)


def make_features(stack, queue):
    """ #11 p13 """
    features = defaultdict(int)
    if len(stack) > 0 and len(queue) > 0:
        features[f'W-1 {stack[-1][1]} W-0 {queue[0][1]}'] = 1
        features[f'W-1 {stack[-1][1]} P-0 {queue[0][2]}'] = 1
        features[f'P-1 {stack[-1][2]} W-0 {queue[0][1]}'] = 1
        features[f'P-1 {stack[-1][2]} P-0 {queue[0][2]}'] = 1
    if len(stack) > 1:
        features[f'W-2 {stack[-2][1]} W-1 {stack[-1][1]}'] = 1
        features[f'W-2 {stack[-2][1]} P-1 {stack[-1][2]}'] = 1
        features[f'P-2 {stack[-2][2]} W-1 {stack[-1][1]}'] = 1
        features[f'P-2 {stack[-2][2]} P-1 {stack[-1][2]}'] = 1
    return features


'''
素性関数と重みを掛け合わせてスコアを計算 
'''
def calc_score(w, feats):
    """ #11 p12 """
    score = [0, 0, 0]
    for i in range(3):
        score[i] = sum(w[i][key] * value for key, value in feats.items())
    return score


'''
@para
queue: [(1,word1,pos1),(2,word2,pos2)...]
w:weight
@return
heads: 各単語の親の ID を格納した配列
'''
def shift_reduce(queue, w):
    """ #11 p15 """
    stack = [(0, 'ROOT', 'ROOT')]
    heads = [-1] * (len(queue) + 1)
    while len(queue) > 0 or len(stack) > 1:
        feats = make_features(stack, queue)
        s = calc_score(w, feats)
        if s[SHIFT] >= s[LEFT] and s[SHIFT] >= s[RIGHT] and len(queue) > 0 \
                or len(stack) < 2:
            stack.append(queue.pop(0))          # shift
        elif s[LEFT] > s[SHIFT] and s[LEFT] >= s[RIGHT]:
            heads[stack[-2][0]] = stack[-1][0]  # reduce 左
            stack.pop(-2)
        else:
            heads[stack[-1][0]] = stack[-2][0]  # reduce 右
            stack.pop(-1)
    return heads


def shift_reduce_train(queue, heads, w):
    """ #11 p17-19 """
    stack = [(0, 'ROOT', 'ROOT')]
    unproc = [heads.count(i) for i in range(len(heads))]
    while len(queue) > 0 or len(stack) > 1:
        feats = make_features(stack, queue)
        s = calc_score(w, feats)
        if s[SHIFT] >= s[LEFT] and s[SHIFT] >= s[RIGHT] and len(queue) > 0 \
                or len(stack) < 2:
            ans = SHIFT
        elif s[LEFT] > s[SHIFT] and s[LEFT] >= s[RIGHT]:
            ans = LEFT
        else:
            ans = RIGHT
        if len(stack) < 2:
            corr = SHIFT
            stack.append(queue.pop(0))
        elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
            # 左が右の親 and 左に未処理の子供がいない
            corr = RIGHT
            unproc[stack[-2][0]] -= 1
            stack.pop(-1)
        elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
            # 右が左の親 and 右に未処理の子供がいない
            corr = LEFT
            unproc[stack[-1][0]] -= 1
            stack.pop(-2)
        else:
            corr = SHIFT
            stack.append(queue.pop(0))
        if ans != corr:
            update_weights(w, feats, ans, corr)

'''
重みの更新
'''
def update_weights(w, feats, ans, corr):
    for key, value in feats.items():
        w[ans][key] -= value
        w[corr][key] += value


def load_mst(path):
    queue = []
    heads = [-1]
    for line in open(path):
        line = line.rstrip()
        if line:
            id, word, _, pos, _, _, head, _ = line.split('\t')
            queue += [(int(id), word, pos)]
            heads += [int(head)]
        else:
            yield queue, heads
            queue = []
            heads = [-1]


def train_sr(train_path, epoch_num=20):
    for _ in tqdm(range(epoch_num)):
        for queue, heads in load_mst(train_path):
            shift_reduce_train(queue, heads, w)


def test_sr(test_path, out_path):
    with open(out_path, 'w') as f_out:
        for queue, _ in load_mst(test_path):
            heads = shift_reduce(queue, w)
            for head in heads[1:]:
                res = '_\t' * 6 + f'{head}\t_'
                print(res, file=f_out)
            print(file=f_out)


if __name__ == '__main__':
    w = [defaultdict(int) for _ in range(3)]
    train_path = '/Users/lingzhidong/Documents/GitHub/nlptutorial/data/mstparser-en-train.dep'
    test_path = '/Users/lingzhidong/Documents/GitHub/nlptutorial/data/mstparser-en-test.dep'
    out_path = './out.txt'
    train_sr(train_path)
    test_sr(test_path, out_path)

    script_path = '/Users/lingzhidong/Documents/GitHub/nlptutorial/script/grade-dep.py'
    subprocess.run(f'python2 {script_path} {test_path} {out_path}'.split())
