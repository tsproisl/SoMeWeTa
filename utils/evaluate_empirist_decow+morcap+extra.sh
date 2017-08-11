#!/bin/bash

# time ../bin/nlp4py-tagger --train tiger+decow+morcap.model --brown /data/decow14_brown.txt --lexicon morphy+capitalization.txt <(cut -f1,3 /cip/corpora/German/Tiger/Tiger2.1/corpus/tiger_release_aug07.txt)
time ../bin/nlp4py-tagger --train tiger+decow+morcap+empirist+extra.model --prior tiger+decow+morcap.model --brown /data/decow14_brown.txt --lexicon morphy+capitalization.txt <(cat ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/train_all_sent_emo.txt additional_training.txt)
../bin/nlp4py-tagger --evaluate tiger+decow+morcap+empirist+extra.model ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/nlp4py-tagger --evaluate tiger+decow+morcap+empirist+extra.model ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/test_web_sent_emo.txt
