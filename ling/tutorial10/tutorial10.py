import math
from collections import defaultdict

# TreePrettyPrinter for visualization
from nltk.tree import Tree
from nltk.treeprettyprinter import TreePrettyPrinter

class cky():
    def __init__(self):
        self.nonterm = []
        self.preterm = defaultdict(list)

    def grammar(self, input_file):
        with open(input_file, "r") as f:
            grammar_file = f.readlines()
        
        for rule in grammar_file:
            lhs, rhs, prob = rule.strip().split("\t")
            rhs_symbols = rhs.split(" ")
            if len(rhs_symbols) == 1:
                self.preterm[rhs].append((lhs, math.log(float(prob))))
            else:
                self.nonterm.append((lhs, rhs_symbols[0], rhs_symbols[1], math.log(float(prob))))

    def cky_parsing(self, input_file):
        with open(input_file, "r") as f:
            data_file = f.readlines()
        
        # output = open("answer.txt", "w")

        for line in data_file:
            self.best_score = defaultdict(lambda : -math.inf)
            self.best_edge = dict()
            self.words = line.strip().split()
            self.wordlen = len(self.words)

            for i in range(self.wordlen):
                for lhs, prob in self.preterm[self.words[i]]:
                    self.best_score[f'{lhs} {i} {i+1}'] = prob

            for j in range(2, self.wordlen+1):
                for i in range(j-2, -1, -1):
                    for k in range(i+1, j):
                        for sym, lsym, rsym, logprob in self.nonterm:
                            if self.best_score[f'{lsym} {i} {k}'] > -math.inf and self.best_score[f'{rsym} {k} {j}'] > -math.inf:
                                my_lp = self.best_score[f'{lsym} {i} {k}'] + self.best_score[f'{rsym} {k} {j}'] + logprob
                                if my_lp > self.best_score[f'{sym} {i} {j}']:
                                    self.best_score[f'{sym} {i} {j}'] = my_lp
                                    self.best_edge[f'{sym} {i} {j}'] = (f'{lsym} {i} {k}', f'{rsym} {k} {j}')
            
            
            # visualization of a parsed tree
            line_s = self.print_tree(f'S 0 {self.wordlen}')
            t = Tree.fromstring(line_s)
            # output.write(TreePrettyPrinter(t).text())
            # output.write("\n" + line_s + "\n")
            print(TreePrettyPrinter(t).text())
            print(line_s)
        # output.close()


    def print_tree(self, sym_i_j):
        sym, i, j = sym_i_j.split(" ")
        i = int(i)
        if sym_i_j in self.best_edge:
            return "(" + sym + " " + self.print_tree(self.best_edge[sym_i_j][0]) + " " + self.print_tree(self.best_edge[sym_i_j][1]) + ")"
        else:
            return "(" + sym + " " + self.words[i] + ")"

if __name__ == '__main__':
    # grammarpath = "/work/test/08-grammar.txt"
    # datapath = "/work/test/08-input.txt"
    grammarpath = r"C:\Users\Lexus\Documents\GitHub\Nlptutorial\data\wiki-en-test.grammar"
    datapath = r"C:\Users\Lexus\Documents\GitHub\Nlptutorial\data\wiki-en-short.tok"

    cky_parse = cky()
    cky_parse.grammar(grammarpath)
    cky_parse.cky_parsing(datapath)