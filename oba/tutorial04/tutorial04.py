from collections import defaultdict
from math import log2
BOS = "<s>"
EOS = "</s>"

class HMM():
    def __init__(self):
        self.context = defaultdict(lambda: 0)
        self.transition = defaultdict(lambda: 0) # 生成を数えるP_E(language|NN) = c(NN→language)/c(NN) →→ P(X|Y)
        self.emit = defaultdict(lambda: 0) # 遷移を数えるP_T(NN|JJ) = c(NN LRB)/c(NN) →→ P(Y)
        self.possible_tags = defaultdict(lambda: 0)
        self.λ = 0.95
        self.N = 1000000

    def load_file(self, file_pth):
        with open(file_pth) as f:
            sentences = [sentence.strip().split() for sentence in f]
            return sentences
    
    def train(self, train_pth):
        sentence = self.load_file(train_pth)
        for word_tags in sentence:
            prev = BOS
            self.context[prev] +=1
            for word_tag in word_tags:
                word, tag = word_tag.split("_")
                self.transition[f"{prev}_{tag}"] += 1 # c(NN LRB)
                self.context[tag] += 1 # c(NN)
                self.emit[f"{tag}_{word}"] += 1 # c(NN→language)
                prev = tag
            self.transition[f"{prev}_{EOS}"] += 1 # BOS JJ ... EOS
        return self
    
    def save(self, save_t_pth, save_e_pth):
        with open(save_t_pth, "w") as t:
            for key, val in self.transition.items():
                prev, tag = key.split("_")
                prob = val/self.context[prev] # c(NN LRB)/c(NN)
                t.write(f"{prev}\t{tag}\t{prob}\n")

        with open(save_e_pth, "w") as e:    
            for key, val in self.emit.items():
                tag, word = key.split("_")
                # 未知語を扱うために平滑化が必要
                prob = self.λ * val/self.context[tag] + (1 - self.λ)/self.N  #c(NN→language)/c(NN)
                e.write(f"{tag}\t{word}\t{prob}\n")
    
    def load(self, t_pth, e_pth):
        with open(t_pth) as t:
            for line in t:
                prev, tag, prob = line.split()
                self.possible_tags[prev] += 1
                self.transition[f"{prev}_{tag}"] = float(prob)
        with open(e_pth) as e:
            for line in e:
                tag, word, prob = line.split()
                self.possible_tags[tag] += 1
                self.emit[f"{tag}_{word}"] = float(prob)

    def smooth(self, p):
        return self.λ * p + (1 - self.λ) / self.N

    def test(self, sentence):
        words = sentence.split()
        # forward
        best_edges = defaultdict(lambda: [0, None]) # [score, edge]
        best_edges[f"0_{BOS}"] = [0, None]
        for i in range(len(words)+1):
            word = words[i] if i < len(words) else EOS
            for prev in self.possible_tags.keys():
                tags = self.possible_tags.keys() if i < len(words) else [EOS]    
                for tag in tags:
                    if f"{i}_{prev}" in best_edges.keys() and f"{prev}_{tag}" in self.transition.keys():
                        # best_score[“i prev”]-logP_T(next|prev)-logP_E(word[i]|next)
                        score = best_edges[f"{i}_{prev}"][0] -\
                                log2(self.transition[f"{prev}_{tag}"]) -\
                                log2(self.smooth(self.emit[f"{tag}_{word}"]))
                        if f"{i+1}_{tag}" not in best_edges:
                            best_edges[f"{i+1}_{tag}"] = [score, f"{i}_{prev}"]
                        if best_edges[f"{i+1}_{tag}"][0] > score:
                            best_edges[f"{i+1}_{tag}"] = [score, f"{i}_{prev}"]
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
    with open(test_file, "r") as sentences:
        for sentence in sentences:
            print(hmm.test(sentence))