from collections import defaultdict
import math


def train_hmm(target): #品詞の遷移確率と単語の生成確率の学習, 入力形式は <単語1>_<品詞> <単語2>_<品詞> ...
    emit = defaultdict(lambda: 0) #品詞タグと単語のペアの数え上げ
    transition = defaultdict(lambda: 0) #品詞の遷移2gramの数え上げ
    context = defaultdict(lambda: 0) #品詞の数え上げ
    f = open("04model.txt", "w")

    for line in target: #数え上げ
        previous = "<s>"
        context[previous] += 1
        line = line.strip()
        words_tags = line.split(" ")

        for word_tag in words_tags:
            l = word_tag.split("_")
            emit["{} {}".format(l[1], l[0])] += 1 #<品詞> <単語>
            transition["{} {}".format(previous, l[1])] += 1
            context[l[1]] += 1
            previous = l[1]
        
        transition["{} </s>".format(previous)] += 1
        context["</s>"] += 1

    transition_list = sorted(transition.items(), key = lambda x: x[0])
    emit_list = sorted(emit.items(), key = lambda x: x[0])

    #遷移確率を出力
    for key, value in transition_list:
        t = key.split(" ") #t[0] = previous, t[1] = word
        f.write("T {} {:.6f}\n".format(key, float(value) / float(context[t[0]])))

    #生成確率を出力
    for key, value in emit_list:
        t = key.split(" ")
        f.write("E {} {:.6f}\n".format(key, float(value) / float(context[t[0]])))

    f.close()


def test_hmm(model, target):
    transition = defaultdict(lambda: 0)
    emission = defaultdict(lambda: 0)
    possible_tags = defaultdict(lambda: 0)
    Lambda = 0.95
    V = 1000000
    f = open("04ans.txt", "w")

    #モデル読み込み
    for line in model:
        line = line.strip()
        prob = line.split(" ") #[<type>, <品詞>, <品詞>or<word>, <prob>]
        possible_tags[prob[1]] = 1 #context[]みたいなもん

        if prob[0] == "T":
            transition["{} {}".format(prob[1], prob[2])] = float(prob[3])
        
        else:
            emission["{} {}".format(prob[1], prob[2])] = float(prob[3])
    
    #前向きステップ
    for line in target:
        line = line.strip()
        words = line.split(" ")
        l = len(words)
        best_score = {}
        best_edge = {}

        best_score["0 <s>"] = 0
        best_edge["0 <s>"] = None

        for i in range(0, l):
            for prev in possible_tags:
                for next in possible_tags:
                    if "{} {}".format(i, prev) in best_score and "{} {}".format(prev, next) in transition:
                        score = best_score["{} {}".format(i, prev)] \
                            - math.log(transition["{} {}".format(prev, next)], 2) \
                            - math.log(emission["{} {}".format(next, words[i])] * Lambda + (1 - Lambda)/V, 2)

                        if "{} {}".format(i + 1, next) not in best_score or best_score["{} {}".format(i + 1, next)] > score:
                            best_score["{} {}".format(i + 1, next)] = score
                            best_edge["{} {}".format(i + 1, next)] = "{} {}".format(i, prev)

        for last in possible_tags:
            if "{} {}".format(l, last) in best_score and "{} {}".format(last, "</s>") in transition:
                score = best_score["{} {}".format(l, last)] \
                    - math.log(transition["{} {}".format(last, "</s>")], 2)
                    
                if "{} {}".format(l + 1, "</s>") not in best_score or best_score["{} {}".format(l + 1, "</s>")] > score:
                    best_score["{} {}".format(l + 1, "</s>")] = score
                    best_edge["{} {}".format(l + 1, "</s>")] = "{} {}".format(l, last)
        
        #後ろむきステップ
        tags = []
        next_edge = best_edge["{} </s>".format(l + 1)]
        
        while next_edge != "0 <s>":
            #このエッジ品詞を出力に追加
            pos_tag = next_edge.split(" ")
            tags.append(pos_tag[1])
            next_edge = best_edge[next_edge]
    
        tags.reverse()
        f.write(" ".join(tags))
        f.write("\n")

    f.close()


#テスト
f1 = open("05-train-input.txt","r")
train_hmm(f1)
f1.close()

f2 = open("04model.txt", "r")
f3 = open("05-test-input.txt", "r")
test_hmm(f2, f3)
f2.close()
f3.close()

#演習
f1 = open("wiki-en-train.norm_pos", "r")
train_hmm(f1)
f1.close()
f2 = open("04model.txt", "r")
f3 = open("wiki-en-test.norm", "r")
test_hmm(f2, f3)
f2.close()
f3.close()

"""
結果

Accuracy: 90.82% (4144/4563)

Most common mistakes:
NNS --> NN      45
NN --> JJ       27
JJ --> DT       22
NNP --> NN      22
VBN --> NN      12
JJ --> NN       12
NN --> IN       11
NN --> DT       10
NNP --> JJ      8
JJ --> RB       7

"""