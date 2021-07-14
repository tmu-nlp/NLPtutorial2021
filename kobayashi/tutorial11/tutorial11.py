from collections import defaultdict

SHIFT, LEFT, RIGHT = 0, 1, 2
 
class DependencyParsing:
    def __init__(self):
        w = [defaultdict(lambda: 0) for _ in range(3)]

    def MakeFeats(self, stack, queue):
        feats = defaultdict(lambda: 0)
        if len(stack) > 0 and len(len) > 0:
            feats[f'W-1 {stack[-1][1]} W-0 {queue[0][1]}'] = 1
            feats[f'W-1 {stack[-1][1]} P-0 {queue[0][2]}'] = 1
            feats[f'P-1 {stack[-1][2]} W-0 {queue[0][1]}'] = 1
            feats[f'P-1 {stack[-1][2]} P-0 {queue[0][2]}'] = 1
        if len(stack) > 1:
            feats[f'W-2 {stack[-2][1]} W-1 {stack[-1][1]}'] = 1
            feats[f'W-2 {stack[-2][1]} P-1 {stack[-1][2]}'] = 1
            feats[f'P-2 {stack[-2][2]} W-1 {stack[-1][1]}'] = 1
            feats[f'P-2 {stack[-2][2]} P-1 {stack[-1][2]}'] = 1
        return feats

    def CalculateScore(self, w, feats):
        score = [0, 0, 0]
        for i in range(3):
            for key, value in feats.items():
                score[i] += w[i][key] * value
        return score

    def ShiftReduce(self, queue):
        heads = [-1] * (len(queue)+1)
        stack = [(0, 'ROOT', 'ROOT')]
        while len(queue) > 0 or len(stack) > 1:
            feats = self.MakeFeats(stack, queue)
            S = self.CalculateScore(self.w, feats)
            if S[SHIFT] >= S[LEFT] and S[SHIFT] >= S[RIGHT] and len(queue) > 0: #shift
                stack.append(queue.pop(0))
            elif S[LEFT] >= S[RIGHT]: #reduce left
                heads[stack[-2][0]] = stack[-1][0]
                stack.pop(-2)
            else: #reduce right
                heads[stack[-1][0]] = stack[-2][0]
                stack.pop(-1)
        return heads

    def UpdateWeights(self, w, feats, ans , corr):
        for key, value, in feats.items():
            self.w[ans][key] -= value
            self.w[corr][key] += value

    def ShiftReduceTrain(self, queue, heads):
        stack = [(0, 'ROOT','ROOT')]
        unproc = []
        for i in range(len(heads)):
            unproc.append(heads.count(i))
        while len(queue) > 0 or len(stack) > 1:
            feats = self.MakeFeats(stack, queue)
            S = self.CalculateScore(self.w, feats)
            if S[SHIFT] >= S[LEFT] and S[SHIFT] >= S[RIGHT] and len(queue) > 0 or len(stack) < 2:
                ans = SHIFT
            elif S[LEFT] > S[SHIFT] and S[LEFT] >= S[RIGHT]:
                ans = LEFT
            else:
                ans = RIGHT

            if len(stack) < 2:
                corr = SHIFT
                stack.append(queue.pop(0))
            elif heads[stack][-1][0] == stack[-2][0] and unproc[stack[-1][0]] == 0:
                # 左が右の親 かつ 左に未処理の子がない
                corr = RIGHT
                unproc[stack[-2][0]] -= 1
                stack.pop(-1)
            elif heads[stack][-2][0] == stack[-1][0] and unproc[stack[-2][0]] == 0:
                # 右が左の親 かつ 右に未処理の子がない
                corr = LEFT
                unproc[stack[-1][0]] -= 1
                stack.pop(-2)
            else:
                corr = SHIFT
                stack.append(queue.pop(0))

            if ans != corr:
                self.UpdateWeights(self.w, feats, ans, corr)

    def train(self, train_file, epoch_num=10):
        for _ in range(epoch_num):
            queue_heads = []
            queue = []
            heads = [-1]
            for line in open(train_file):
                line = line.strip().split('\t')
                if line:
                    queue.append((int(line[0]), line[1], line[3]))
                    heads.append(int(line[6]))
                else:
                    yield queue, heads
                    queue = []
                    heads = [-1]
                queue_heads.append(queue, heads)
            for queue, heads in queue_heads:
                self.ShiftReduceTrain(queue, heads)

    def test(self, test_file, output_file):
        with open(output_file, 'w') as out_f:
            queues = []
            queue = []
            for line in open(test_file):
                line = line.strip().split('\t')
                if line:
                    queue.append((int(line[0]), line[1], line[3]))
                else:
                    yield queue
                    queue = []
            for queue in queues:
                heads = self.ShiftReduce(queue)
                for head in heads[1:]:
                    out_f.write('_\t'*6 + f'{head}\t_\n')
            
if __name__ == '__main__':
    train_file = '../data/mstparser-en-train.dep'
    test_file = '../data/mstparser-en-test.dep'
    output_file = 'out.txt'
    parser = DependencyParsing()
    parser.train(train_file)
    parser.test(test_file, output_file)