
from collections import defaultdict
from math import log10, inf
from tqdm import tqdm

class CKY():
    def __init__(self) -> None:
        self.preterm = defaultdict(list)
        self.nonterm = []
        self.words_per_sentece = []
        self.best_score = defaultdict(lambda: -inf)
        self.best_edge = []

    def load_grammar(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
        for grammar in lines:
            lhs, rhs, prob = grammar.split("\t")
            prob = float(prob)
            splited_rhs = rhs.split()
            if len(splited_rhs) == 1: # 全終端記号
                self.preterm[rhs].append((lhs, log10(prob)))
            else:
                self.nonterm.append((lhs, splited_rhs[0], splited_rhs[1], log10(prob)))
    
    def load_sentence(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            self.words_per_sentece = [line.strip().split() for line in lines]
    
    def add_preterm(self):
        for words_idx in range(len(self.words_per_sentece)):
            for word_idx in range(len(self.words_per_sentece[words_idx])):
                if self.preterm[self.words_per_sentece[words_idx][word_idx]]:
                    for lhs, log_prob in self.preterm[self.words_per_sentece[words_idx][word_idx]]:
                        self.best_score[f"{lhs}_{word_idx}_{word_idx+1}"] = log_prob
    
    def combine_nonterm(self):
        for words_idx in range(len(self.words_per_sentece)):
            for r in range(2, len(self.words_per_sentece[words_idx])+1): # スパンの右側
                for l in reversed(range(0, r-2)): # スパンの左側
                    for st in range(l+1, r): # rsym の開始点
                        for sym, lsym, rsym, log_prob in self.nonterm:
                            if self.best_score[f"{lsym}_{l}_{st}"] > -inf and self.best_score[f"{rsym}_{st}_{r}"] > -inf:
                                my_lp = self.best_score[f"{lsym}_{l}_{st}"] + self.best_score[f"{rsym}_{st}_{r}"] + log_prob
                                if my_lp > self.best_score[f"{sym}_{l}_{r}"]:
                                    self.best_score[f"{sym}_{l}_{r}"] = my_lp
                                    self.best_edge[f"{sym}_{l}_{r}"] = (f"{lsym}_{l}_{st}", f"{rsym}_{st}_{r}")
            print(self.print_tree(words_idx, f"S(0, {len(self.words_per_sentece[words_idx])})"))
    
    def print_tree(self, words_idx, sym):
        if sym in self.best_edge:
            return f"({sym} {self.print_tree(self.best_edge[0])} {self.print_tree(self.best_edge[1])})"
        else:
            return f"({sym} {self.words_per_sentece[words_idx]})"

    def main(self, grammar_file_path, input_file_path):
        self.load_grammar(grammar_file_path)
        self.load_sentence(input_file_path)
        self.add_preterm()
        self.combine_nonterm()

if __name__ == "__main__":
    grammar_file = "data/wiki­-en-­test.grammar"
    input_file = "data/wiki-­en-­short.tok"
    grammar_file = grammar_file.replace("\xad", "")
    input_file = input_file.replace("\xad", "")

    model = CKY()
    model.main(grammar_file, input_file)