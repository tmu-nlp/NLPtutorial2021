from collections import defaultdict
import numpy as np
import re

class CKY:
    def __init__(self):
        self.nonterm = []
        self.preterm = defaultdict(list)

    def load_grammar(self, grammar_file):
        with open(grammar_file) as f:
            for rule in f:
                lhs, rhs, prob = rule.strip().split('\t')
                rhs_symbol = rhs.split()
                prob = float(prob)
                if len(rhs_symbol) == 1:
                    self.preterm[rhs].append((lhs, np.log(prob)))
                else:
                    self.nonterm.append((lhs, rhs_symbol[0], rhs_symbol[1], np.log(prob)))
    
    def print_tree(self, sym_ij):
        sym, i = re.sub(r"(.*?)\((.+?),\s(.+?)\)", r"\1 \2", sym_ij).split()
        i = int(i)
        if sym_ij in self.best_edge.keys():
            return "(" + sym + " " \
                    + self.print_tree(self.best_edge[sym_ij][0]) + " " \
                    + self.print_tree(self.best_edge[sym_ij][1]) + ")"
        else:
            return "(" + sym + " " + self.words[i] + ")"

    def main(self, input_file):
        with open(input_file) as f:
            for line in f:
                self.words = line.strip().split()
                self.best_score = defaultdict(lambda: -float('inf'))
                self.best_edge = {}
                for i in range(len(self.words)):
                    if self.preterm[self.words[i]] != []:
                        for lhs, log_prob in self.preterm[self.words[i]]:
                            self.best_score[f"{lhs}({i}, {i+1})"] = log_prob
                for j in range(2, len(self.words)+1):
                    for i in range(j-2, -1, -1):
                        for k in range(i+1, j):
                            for sym, lsym, rsym, logprob in self.nonterm:
                                if self.best_score[f"{lsym}({i}, {k})"] > -float('inf') and self.best_score[f"{rsym}({k}, {j})"] > -float('inf'):
                                    my_lp = self.best_score[f"{lsym}({i}, {k})"] + self.best_score[f"{rsym}({k}, {j})"] + logprob
                                    if my_lp > self.best_score[f"{sym}({i}, {j})"]:
                                        self.best_score[f"{sym}({i}, {j})"] = my_lp
                                        self.best_edge[f"{sym}({i}, {j})"] = (f"{lsym}({i}, {k})", f"{rsym}({k}, {j})")
                
                print(self.print_tree(f"S(0, {len(self.words)})"))

    
if __name__ == '__main__':
    cky = CKY()
    grammar_file = '../data/wiki-en-test.grammar'
    cky.load_grammar(grammar_file)
    input_file = '../data/wiki-en-short.tok'
    cky.main(input_file)

'''
(S (NP_PRP i) (VP (VBD saw) (VP' (NP (DT a) (NN girl)) (PP (IN with) (NP (DT a) (NN telescope))))))
(nlp_intro) kobayashimasamune@gvpn162196 tutorial10 % python tutorial10.py
(S (PP (IN Among) (NP (DT these) (NP' (, ,) (NP' (JJ supervised) (NP' (NN learning) (NNS approaches)))))) (S' (VP (VBP have) (VP (VBN been) (VP' (NP (DT the) (NP' (ADJP (RBS most) (JJ successful)) (NNS algorithms))) (PP (TO to) (NP_NN date))))) (. .)))
(S (NP (JJ Current) (NN accuracy)) (S' (VP (VBZ is) (ADJP (JJ difficult) (S_VP (TO to) (VP (VB state) (PP (IN without) (NP (NP (DT a) (NN host)) (PP (IN of) (NP_NNS caveats)))))))) (. .)))
(S (NP (NNP WSD) (NN task)) (S' (VP (VBZ has) (NP (NP (NP (CD two) (NNS variants)) (NP' (: :) (NP' (`` ``) (NP' (JJ lexical) (NP' (NN sample) ('' '')))))) (NP' (CC and) (NP (`` ``) (NP' (PDT all) (NP' (NNS words) (NP' ('' '') (NN task)))))))) (. .)))
(S (NP (NP (DT The) (NP' (NN bass) (NN line))) (PP (IN of) (NP (DT the) (NN song)))) (S' (VP (VBZ is) (ADJP (RB too) (JJ weak))) (. .)))
(S (NP (JJ Early) (NNS researchers)) (S' (VP (VBD understood) (VP' (NP (NP (DT the) (NP' (NN significance) (NP' (CC and) (NN difficulty)))) (PP (IN of) (NP_NNP WSD))) (ADVP_RB well))) (. .)))
(S (ADVP_RB Still) (S' (, ,) (S' (NP (JJ supervised) (NNS systems)) (S' (VP (VBP continue) (S_VP (TO to) (VP (VB perform) (ADVP_RBS best)))) (. .)))))
(S (NP (NP (JJ Difficulties) (NNS Differences)) (PP (IN between) (NP_NNS dictionaries))) (S' (NP (NP (CD One) (NN problem)) (PP (IN with) (NP (NN word) (NP' (NN sense) (NN disambiguation))))) (S' (VP (VBZ is) (VP (VBG deciding) (SBAR (WHNP_WP what) (S (NP (DT the) (NNS senses)) (VP_VBP are))))) (. .))))
(S (PP (IN In) (NP (NP_NNS cases) (PP (IN like) (NP (NP (DT the) (NP' (NN word) (NN bass))) (ADVP_IN above))))) (S' (, ,) (S' (NP (QP (IN at) (QP' (JJS least) (DT some))) (NNS senses)) (S' (VP (VBP are) (ADJP (RB obviously) (JJ different))) (. .)))))
(S (NP (NP (JJ Different) (NNS dictionaries)) (NP' (CC and) (NNS thesauruses))) (S' (VP (MD will) (VP (VB provide) (NP (NP (JJ different) (NNS divisions)) (PP (IN of) (NP (NP_NNS words) (PP (IN into) (NP_NNS senses))))))) (. .)))
(S (S (NP (JJ Other) (NNS resources)) (VP (VBN used) (PP (IN for) (NP (NN disambiguation) (NNS purposes))))) (S' (VP (VBP include) (NP (NP (NNP Roget) (POS 's)) (NP' (NNS Thesaurus) (NP' (CC and) (NP_NNP Wikipedia))))) (. .)))
(S (NP_PRP It) (S' (VP (VBZ is) (ADJP (JJ instructive) (S_VP (TO to) (VP (VB compare) (NP (NP (NP (DT the) (NP' (NN word) (NN sense))) (NP' (NN disambiguation) (NN problem))) (PP (IN with) (NP (NP (DT the) (NN problem)) (PP (IN of) (NP (JJ part-of-speech) (NN tagging)))))))))) (. .)))
(S (S (NP_DT Both) (VP (VBP involve) (S_VP (VBG disambiguating) (VP' (CC or) (VP (VBG tagging) (PP (IN with) (NP_NNS words))))))) (S' (, ,) (S' (VP (VB be) (VP' (NP_PRP it) (PP (IN with) (NP (NP (NNS senses) (NP' (CC or) (NNS parts))) (PP (IN of) (NP_NN speech)))))) (. .))))
(S (S (NP (DT These) (NNS figures)) (VP (VBP are) (ADJP (JJ typical) (PP (IN for) (NP_NNP English))))) (S' (, ,) (S' (CC and) (S' (VP (MD may) (VP (VB be) (VP' (ADJP (RB very) (JJ different)) (PP (IN from) (NP (NP_DT those) (PP (IN for) (NP (JJ other) (NNS languages)))))))) (. .)))))
(S (S (NP_NNP Inter-judge) (VP (VBP variance) (NP (DT Another) (NN problem)))) (S' (VP (VBZ is) (NP (JJ inter-judge) (NN variance))) (. .)))
(S (S (NP (NNP WSD) (NNS systems)) (VP (VBP are) (VP' (ADVP_RB normally) (VP (VBN tested) (PP (IN by) (S_VP (VBG having) (NP (NP (PRP$ their) (NNS results)) (PP (IN on) (NP (DT a) (NN task)))))))))) (S' (VP (VBN compared) (PP (IN against) (NP (NP_DT those) (PP (IN of) (NP (DT a) (NN human)))))) (. .)))
(S (`` ``) (S' (VP (VP_VBZ Jill) (VP' (CC and) (VP_SBAR (SBAR (S (NP_NNP Mary) (VP (VBP are) (NP_NNS mothers))) (SBAR' (. .) ('' ''))) (SBAR' (: --) (PRN (-LRB- -LRB-) (PRN' (S (NP_DT each) (VP (VBZ is) (VP' (ADVP_RB independently) (NP (DT a) (NN mother))))) (-RRB- -RRB-))))))) (. .)))
(S (S_VP (TO To) (VP (ADVP_RB properly) (VP' (VB identify) (NP (NP_NNS senses) (PP (IN of) (NP_NNS words)))))) (S' (NP_PRP one) (S' (VP (MD must) (VP (VB know) (NP (JJ common) (NP' (NN sense) (NNS facts))))) (. .))))
(S (ADVP_RB Also) (S' (, ,) (S' (NP (ADJP (RB completely) (JJ different)) (NNS algorithms)) (S' (VP (MD might) (VP (VB be) (VP (VBN required) (PP (IN by) (NP (JJ different) (NNS applications)))))) (. .)))))
(S (PP (IN In) (NP (NN machine) (NN translation))) (S' (, ,) (S' (NP (DT the) (NN problem)) (S' (VP (VBZ takes) (NP (NP (DT the) (NN form)) (PP (IN of) (NP (NN target) (NP' (NN word) (NN selection)))))) (. .)))))
(S (S (NP (NP_NNS Discreteness) (PP (IN of) (NP_NNS senses))) (VP (ADVP_RB Finally) (VP' (, ,) (VP' (NP (DT the) (NP' (JJ very) (NN notion))) (PP (IN of) (NP (`` ``) (NP' (NN word) (NP' (NN sense) ('' ''))))))))) (S' (VP (VBZ is) (ADJP (JJ slippery) (ADJP' (CC and) (JJ controversial)))) (. .)))
(S (NP (JJ Word) (NN meaning)) (S' (VP (VBZ is) (VP' (PP (IN in) (NP (NP (NN principle) (NP' (NN infinitely) (NN variable))) (NP' (CC and) (NN context)))) (ADJP_JJ sensitive))) (. .)))
(S (NP_PRP It) (S' (VP (VBZ does) (VP' (RB not) (VP (VB divide) (VP' (PRT_RP up) (VP' (ADVP_RB easily) (PP (IN into) (NP (JJ distinct) (NP' (CC or) (NP (JJ discrete) (NNS sub-meanings)))))))))) (. .)))
(S (NP_NNP Deep) (S' (VP (VBZ approaches) (NP (NP (JJ presume) (NN access)) (PP (TO to) (NP (NP (DT a) (NP' (JJ comprehensive) (NN body))) (PP (IN of) (NP (NN world) (NN knowledge))))))) (. .)))
(S (NP (NNP Shallow) (NNS approaches)) (S' (VP (VBP do) (VP' (RB n't) (VP (VB try) (S_VP (TO to) (VP (VB understand) (NP (DT the) (NN text))))))) (. .)))
(S (NP (JJ Supervised) (NNS methods)) (S' (: :) (S' (S (NP_DT These) (VP (VBP make) (VP (VBP use) (VP' (PP (IN of) (NP (JJ sense-annotated) (NN corpora))) (S_VP (TO to) (VP (VB train) (PP_IN from))))))) (. .))))
(S (NP (JJ Unsupervised) (NNS methods)) (S' (: :) (S' (NP (DT These) (NP' (ADJP (ADJP_JJ eschew) (PRN (-LRB- -LRB-) (PRN' (ADVP_RB almost) (-RRB- -RRB-)))) (NP' (ADJP (RB completely) (JJ external)) (NN information)))) (S' (CC and) (S' (VP (VB work) (VP' (ADVP_RB directly) (PP (IN from) (NP (JJ raw) (NP' (JJ unannotated) (NN corpora)))))) (. .))))))
(S (NP (DT These) (NNS methods)) (VP (VBP are) (S (ADVP_RB also) (S' (VP (VBN known) (PP (IN under) (NP (NP (DT the) (NN name)) (PP (IN of) (NP (NN word) (NP' (NN sense) (NN discrimination))))))) (. .)))))
(S (S (NP (CD Two) (NP' (JJ shallow) (NNS approaches))) (VP (VBN used) (S_VP (TO to) (VP (VB train) (VP' (CC and) (VP' (ADVP_RB then) (VP_VB disambiguate))))))) (S' (VP (VBP are) (NP (NP (NNP Na??ve) (NNP Bayes)) (NP' (NNS classifiers) (NP' (CC and) (NP (NN decision) (NNS trees)))))) (. .)))
(S (PP (IN In) (NP (JJ recent) (NN research))) (S' (, ,) (S' (NP (NP (JJ kernel-based) (NNS methods)) (PP (JJ such) (PP' (IN as) (NP (NN support) (NP' (NN vector) (NNS machines)))))) (S' (VP (VBP have) (VP (VBN shown) (VP' (NP (JJ superior) (NN performance)) (PP (IN in) (NP (JJ supervised) (NN learning)))))) (. .)))))
(S (NP_NNP Dictionary) (S' (: -) (S' (CC and) (S' (NP (JJ knowledge-based) (NNS methods)) (S' (NP (DT The) (NP' (NNP Lesk) (NN algorithm))) (S' (VP (VBZ is) (NP (DT the) (NP' (JJ seminal) (NP' (JJ dictionary-based) (NN method))))) (. .)))))))
(S (NP (DT The) (NP' (NNP Yarowsky) (NN algorithm))) (VP (VBD was) (VP' (NP (DT an) (NP' (JJ early) (NN example))) (PP (IN of) (NP (PDT such) (NP' (NP (DT an) (NN algorithm)) (. .)))))))
(S (S (NP (DT The) (NNS seeds)) (VP (VBP are) (VP (VBN used) (S_VP (TO to) (VP (VB train) (NP (DT an) (NP' (JJ initial) (NN classifier)))))))) (S' (, ,) (S' (VP (VBG using) (NP (DT any) (NP' (JJ supervised) (NN method)))) (. .))))
(S (S (NP (JJ Other) (NP' (JJ semi-supervised) (NNS techniques))) (VP (VBP use) (NP (NP (JJ large) (NNS quantities)) (PP (IN of) (NP (JJ untagged) (NN corpora)))))) (S' (VP (TO to) (VP (VB provide) (VP' (NP (JJ co-occurrence) (NN information)) (SBAR (WHNP_WDT that) (S_VP (VBZ supplements) (NP (DT the) (NP' (JJ tagged) (NN corpora)))))))) (. .)))
(S (S (NP (DT These) (NNS techniques)) (VP (VBP have) (NP (DT the) (NN potential)))) (S' (VP (TO to) (VP (VB help) (PP (IN in) (NP (NP (DT the) (NN adaptation)) (PP (IN of) (NP (NP (JJ supervised) (NNS models)) (PP (TO to) (NP (JJ different) (NNS domains))))))))) (. .)))
(S (NP (JJ Word-aligned) (NP' (JJ bilingual) (NN corpora))) (S' (VP (VBP have) (VP (VBN been) (VP (VBN used) (S_VP (TO to) (VP (VB infer) (VP' (NP (JJ cross-lingual) (NP' (NN sense) (NNS distinctions))) (VP' (, ,) (VP' (NP (DT a) (NN kind)) (PP (IN of) (NP (JJ semi-supervised) (NN system))))))))))) (. .)))
(S (S (NP (JJ Unsupervised) (NNS methods)) (VP (VBZ Main) (NP (NP (NP_NN article) (NP' (: :) (NP' (JJ Word) (NN sense)))) (NP' (NN induction) (NP' (JJ Unsupervised) (NN learning)))))) (S' (VP (VBZ is) (NP (NP (DT the) (NP' (JJS greatest) (NN challenge))) (PP (IN for) (NP (NNP WSD) (NNS researchers))))) (. .)))
(S (ADVP_RB Then) (S' (, ,) (S' (NP (NP (JJ new) (NNS occurrences)) (PP (IN of) (NP (DT the) (NN word)))) (S' (VP (MD can) (VP (VB be) (VP (VBN classified) (PP (IN into) (NP (DT the) (NP' (JJS closest) (NP' (CD induced) (NNS clusters\/senses)))))))) (. .)))))
(S (ADVP_RB Alternatively) (S' (, ,) (S' (NP (NN word) (NP' (NN sense) (NP' (NN induction) (NNS methods)))) (S' (VP (MD can) (VP (VB be) (VP (VBN tested) (VP' (CC and) (VP (VBN compared) (PP (IN within) (NP (DT an) (NN application)))))))) (. .)))))
(S (NP (NP (JJ Local) (NNS impediments)) (NP' (CC and) (NP_NN summary))) (S' (NP (DT The) (NP' (NN knowledge) (NP' (NN acquisition) (NN bottleneck)))) (S' (VP (VBZ is) (VP' (ADVP_RB perhaps) (VP' (NP (DT the) (NP' (JJ major) (NN impediment))) (S_VP (TO to) (VP (VBG solving) (NP (DT the) (NP' (NNP WSD) (NN problem)))))))) (. .))))
(S (NP (JJ Unsupervised) (NNS methods)) (S' (VP (VBP rely) (VP' (PP (IN on) (NP (NP_NN knowledge) (PP (IN about) (NP (NN word) (NNS senses))))) (VP' (, ,) (SBAR (WHNP_WDT which) (S_VP (VBZ is) (VP' (ADVP_RB barely) (VP (VBN formulated) (PP (IN in) (NP (JJ dictionaries) (NP' (CC and) (NP (JJ lexical) (NNS databases)))))))))))) (. .)))
(S (NP (JJ Knowledge) (NNS sources)) (S' (VP (VB provide) (NP (NP_NNS data) (SBAR (WHNP_WDT which) (S_VP (VBP are) (ADJP (JJ essential) (PP (TO to) (NP (NP (JJ associate) (NNS senses)) (PP (IN with) (NP_NNS words))))))))) (. .)))
(S (SBAR (IN In) (SBAR' (NN order) (S_VP (TO to) (VP (VB test) (NP (NP (CD one) (POS 's)) (NP' (NN algorithm) (NP' (, ,) (NP_NNS developers)))))))) (S' (VP (MD should) (VP (VB spend) (VP' (NP (PRP$ their) (NN time)) (S_VP (TO to) (VP (VB annotate) (NP (PDT all) (NP' (NN word) (NNS occurrences)))))))) (. .)))
(S (CC And) (S' (VP (VB comparing) (VP' (NP_NNS methods) (VP' (ADVP_RB even) (SBAR (IN on) (S (NP (DT the) (NP' (JJ same) (NN corpus))) (VP (VBZ is) (VP' (RB not) (VP' (ADJP_JJ eligible) (SBAR (IN if) (S (NP_EX there) (VP (VBZ is) (NP (JJ different) (NP' (NN sense) (NNS inventories)))))))))))))) (. .)))
(S (SBAR (IN In) (SBAR' (NN order) (S_VP (TO to) (VP (VB define) (NP (NP (JJ common) (NP' (NN evaluation) (NNS datasets))) (NP' (CC and) (NNS procedures))))))) (S' (, ,) (S' (NP (JJ public) (NP' (NN evaluation) (NNS campaigns))) (S' (VP (VBP have) (VP (VBN been) (VP_VBN organized))) (. .)))))
(S (NP (NNP Task) (NNP Design)) (S' (VP (VBZ Choices) (NP (NN Sense) (NNS Inventories))) (. .)))
(S (PP (IN During) (NP (DT the) (NP' (JJ first) (NP' (NNP Senseval) (NN workshop))))) (S' (NP (DT the) (NP' (JJ HECTOR) (NP' (NN sense) (NN inventory)))) (S' (VP (VBD was) (VP_VBN adopted)) (. .))))
(S (NP (DT A) (NN set)) (S' (IN of) (S' (VP (VBG testing) (NP_NNS words)) (. .))))
(S (NP_NNP Comparison) (S' (PP (IN of) (NP_NNS methods)) (S' (VP (MD can) (VP (VB be) (VP (VBN divided) (VP' (PP (IN in) (NP (NP (CD 2) (NNS groups)) (PP (IN by) (NP (NP_NN amount) (PP (IN of) (NP_NNS words)))))) (S_VP (TO to) (VP_VB test)))))) (. .))))
(S (ADVP_RB Initially) (S' (ADVP_RB only) (S' (S (NP (DT the) (NN latter)) (VP (VBD was) (VP (VBN used) (PP (IN in) (NP_NN evaluation))))) (S' (CC but) (S' (ADVP_RB later) (S' (NP (DT the) (JJ former)) (S' (VP (VBD was) (VP_VBN included)) (. .))))))))
(S (NP (JJ Lexical) (NP' (NN sample) (NNS organizers))) (VP (VBD had) (S_VP (TO to) (VP (VB choose) (NP (NP_NNS samples) (SBAR (WHPP (IN on) (WHNP_WDT which)) (S (NP (DT the) (NNS systems)) (S' (VP (VBD were) (S_VP (TO to) (VP (VB be) (VP_VBN tested)))) (. .)))))))))
(S Baselines)
(S (PP (IN For) (NP (NN comparison) (NNS purposes))) (S' (, ,) (S' (NP (ADJP (VBN known) (ADJP' (, ,) (ADJP' (CC yet) (JJ simple)))) (NP' (, ,) (NP_NNS algorithms))) (S' (VP (VBD named) (SBAR_S (NP_NNS baselines) (VP (VBP are) (VP_VBN used)))) (. .)))))
(S (NP_DT These) (S' (VP (VBP include) (NP (NP (JJ different) (NNS variants)) (PP (IN of) (NP (NNP Lesk) (NP' (UCP (NP_NN algorithm) (UCP' (CC or) (ADJP (RBS most) (JJ frequent)))) (NP' (NN sense) (NN algorithm))))))) (. .)))
(S Sense)
(S (NP_NNP WordNet) (S' (VP (VBZ is) (NP (NP (DT the) (NP' (ADJP (RBS most) (JJ popular)) (NN example))) (PP (IN of) (NP (NN sense) (NN inventory))))) (. .)))
(S (S (NP (NP (DT The) (NN reason)) (PP (IN for) (S_VP (VBG adopting) (NP (NP (DT the) (NP' (JJ HECTOR) (NN database))) (PP (IN during) (NP_NNS Senseval-1)))))) (VP (VBD was) (PP (IN that) (NP (DT the) (NP' (NNP WordNet) (NN inventory)))))) (S' (VP (VBD was) (VP' (ADVP_RB already) (ADJP (RB publicly) (JJ available)))) (. .)))
(S (NP_NNP Evaluation) (S' (VP_VBZ measures) (. .)))
'''