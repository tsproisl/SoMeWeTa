#!/bin/bash

time ../bin/nlp4py-tagger --train tiger+dewikiall+morphy.model --brown /data/dewiki_all_brown.txt --lexicon /cip/corpora/Lexicon/Morphy/morphy_wc.txt <(cut -f1,3 /cip/corpora/German/Tiger/Tiger2.1/corpus/tiger_release_aug07.txt)
time ../bin/nlp4py-tagger --train tiger+dewikiall+morphy+empirist.model --prior tiger+dewikiall+morphy.model --brown /data/dewiki_all_brown.txt --lexicon /cip/corpora/Lexicon/Morphy/morphy_wc.txt ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/train_all_sent.txt
../bin/nlp4py-tagger --evaluate tiger+dewikiall+morphy+empirist.model ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/test_cmc_sent.txt
../bin/nlp4py-tagger --evaluate tiger+dewikiall+morphy+empirist.model ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/test_web_sent.txt
