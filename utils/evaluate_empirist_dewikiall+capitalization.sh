#!/bin/bash

time ../bin/nlp4py-tagger --train tiger+dewikiall+capit.model --brown /data/dewiki_all_brown.txt --lexicon /cip/corpora/Web/dewiki/dewiki-20170620_all_capitalization_features.txt <(cut -f1,3 /cip/corpora/German/Tiger/Tiger2.1/corpus/tiger_release_aug07.txt)
time ../bin/nlp4py-tagger --train tiger+dewikiall+capit+empirist.model --prior tiger+dewikiall+capit.model --brown /data/dewiki_all_brown.txt --lexicon /cip/corpora/Web/dewiki/dewiki-20170620_all_capitalization_features.txt ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/train_all_sent_emo.txt
../bin/nlp4py-tagger --evaluate tiger+dewikiall+capit+empirist.model ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/nlp4py-tagger --evaluate tiger+dewikiall+capit+empirist.model ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/test_web_sent_emo.txt
