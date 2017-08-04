#!/bin/bash

time ../bin/nlp4py-tagger --train tiger+decow.model --brown /data/decow14_brown.txt <(cut -f1,3 /cip/corpora/German/Tiger/Tiger2.1/corpus/tiger_release_aug07.txt)
time ../bin/nlp4py-tagger --train tiger+decow+empirist.model --prior tiger+decow.model --brown /data/decow14_brown.txt ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/train_all_sent.txt
../bin/nlp4py-tagger --evaluate tiger+decow+empirist.model ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/test_cmc_sent.txt
../bin/nlp4py-tagger --evaluate tiger+decow+empirist.model ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/test_web_sent.txt
