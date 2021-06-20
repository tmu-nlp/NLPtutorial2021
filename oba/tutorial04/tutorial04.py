from collections import defaultdict
from math import log2
BOS = "<s>"
EOS = "</s>"

class HMM():
    def __init__(self):
        self.context = defaultdict(lambda: 0) # {タグ：訓練データに出現した回数} c(NN)
        self.transition = defaultdict(lambda: 0) # {タグの遷移：出現回数} c(NN LRB)
        self.emit = defaultdict(lambda: 0) # {単語の生成：出現回数} c(NN→language)
        self.possible_tags = defaultdict(lambda: 0)
        self.λ = 0.95
        self.λ_unk = 1 - self.λ
        self.N = 1000000

    def load_file(self, file_pth):
        with open(file_pth) as f:
            sentences = [sentence.strip().split() for sentence in f]
            return sentences
    
    def train(self, train_pth):
        sentence = self.load_file(train_pth)
        for word_tags in sentence:
            prev_tag = BOS
            self.context[prev_tag] += 1
            for word_tag in word_tags:
                word, tag = word_tag.split("_")
                self.transition[f"{prev_tag}_{tag}"] += 1
                self.context[tag] += 1
                self.emit[f"{tag}_{word}"] += 1
                prev_tag = tag
            self.transition[f"{prev_tag}_{EOS}"] += 1
        return self
    
    def save(self, save_t_pth, save_e_pth):
        with open(save_t_pth, "w") as t:
            for tags, count in self.transition.items():
                prev_tag, tag = tags.split("_")
                prob = count / self.context[prev_tag] # P(LRB|NN)=c(NN LRB)/c(NN)
                t.write(f"{prev_tag}\t{tag}\t{prob}\n")

        with open(save_e_pth, "w") as e:    
            for tag_word, count in self.emit.items():
                tag, word = tag_word.split("_")
                prob = count / self.context[tag] #c(NN→language)/c(NN)
                # 未知語を扱うために平滑化が必要
                smoothed_prob = self.smooth(prob)
                e.write(f"{tag}\t{word}\t{smoothed_prob}\n")
    
    def load(self, t_pth, e_pth):
        with open(t_pth) as t:
            for line in t:
                prev_tag, tag, prob = line.split()
                self.possible_tags[prev_tag] += 1
                self.transition[f"{prev_tag}_{tag}"] = float(prob)
        with open(e_pth) as e:
            for line in e:
                tag, word, prob = line.split()
                self.possible_tags[tag] += 1
                self.emit[f"{tag}_{word}"] = float(prob)

    def smooth(self, p):
        return self.λ*p + self.λ_unk/self.N

    def test(self, sentence):
        words = sentence.split()
        # forward
        best_edges = defaultdict(lambda: [0, None]) # {idx_tag:[score, idx_prevtag]}
        best_edges[f"0_{BOS}"] = [0, None]
        for i in range(len(words)+1):
            word = words[i] if i < len(words) else EOS
            for prev_tag in self.possible_tags.keys():
                tags = self.possible_tags.keys() if i < len(words) else [EOS]    
                for tag in tags:
                    if f"{i}_{prev_tag}" in best_edges.keys() and f"{prev_tag}_{tag}" in self.transition.keys():
                        # best_score[“i prev”]-logP_T(next|prev)-logP_E(word[i]|next)
                        score = best_edges[f"{i}_{prev_tag}"][0] -\
                                log2(self.transition[f"{prev_tag}_{tag}"]) -\
                                log2(self.smooth(self.emit[f"{tag}_{word}"]))
                        if f"{i+1}_{tag}" not in best_edges:
                            best_edges[f"{i+1}_{tag}"] = [score, f"{i}_{prev_tag}"]
                        if best_edges[f"{i+1}_{tag}"][0] > score:
                            best_edges[f"{i+1}_{tag}"] = [score, f"{i}_{prev_tag}"]
        # backward
        tags = []
        next_edge = best_edges[f"{len(words)+1}_{EOS}"][1]
        while next_edge[0] != '0':
            idx, tag = next_edge.split("_")
            idx = int(idx)
            tags.append(tag)
            next_edge = best_edges[next_edge][1]
        return " ".join(list(reversed(tags)))           

if __name__ == "__main__":
    hmm = HMM()
    train_file = "../data/wiki-en-train.norm_pos"
    save_t_file = "tutorial04_t.txt"
    save_e_file = "tutorial04_e.txt"
    hmm.train(train_file).save(save_t_file, save_e_file)
    hmm.load(save_t_file, save_e_file)
    
    test_file = "../data/wiki-en-test.norm"
    with open(test_file, "r") as f:
        sentences = f.readlines()
    ans_file = "tutorial04_ans.txt"
    with open(ans_file, "w") as f:
        for sentence in sentences:
            f.write(hmm.test(sentence)+"\n")

'''
Accuracy: 90.82% (4144/4563)

Most common mistakes:
NNS --> NN      45
NN --> JJ       27
NNP --> NN      22
JJ --> DT       22
JJ --> NN       12
VBN --> NN      12
NN --> IN       11
NN --> DT       10
NNP --> JJ      8
JJ --> VBN      7
'''