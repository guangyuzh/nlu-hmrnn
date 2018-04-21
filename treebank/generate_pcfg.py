from nltk.corpus import treebank
from nltk import induce_pcfg
from nltk.parse import pchart
from nltk.parse import ViterbiParser
from nltk import Nonterminal
import sys

# Reference: https://www.cs.bgu.ac.il/~elhadad/nlp16/NLTK-PCFG.html


class TreebankPCFG():
    """
    Generate PCFG from 10% Penn Treebank
    """
    def __init__(self):
        self._induce_grammar()
        self._induce_pcfg()

    def _induce_grammar(self):
        self.productions = []
        for tree in treebank.parsed_sents(treebank.fileids()):
            # perform optional tree transformations, e.g.:
            tree.collapse_unary(collapsePOS=False)  # Remove branches A-B-C into A-B+C
            tree.chomsky_normal_form(horzMarkov=2)  # Remove A->(B,C,D) into A->B,C+D->D
            self.productions += tree.productions()

    def _induce_pcfg(self):
        S = Nonterminal('S')
        self.grammar = induce_pcfg(S, self.productions)

    def parse_sentence(self, sent):
        """
        Parse sent using induced grammar
        Visualize the most likely parse tree for sent
        :return: None. Save parsing results to pcfg.txt
        """
        if self.grammar is None:
            raise ValueError("PCFG hasn't been induced yet.")
        # other parser option(s): e.g., parser = pchart.InsideChartParser(self.grammar)
        parser = ViterbiParser(self.grammar)
        parser.trace(3)

        # http://www.nltk.org/api/nltk.parse.html
        sys.stdout = open('pcfg.txt', 'w')
        parses = parser.parse(sent)
        for parse in parses:
            print(parse)
            # visualize the tree:
            print(parse.draw())


if __name__ == "__main__":
    pcfg = TreebankPCFG()

    sent = treebank.parsed_sents('wsj_0001.mrg')[0].leaves()
    pcfg.parse_sentence(sent)
