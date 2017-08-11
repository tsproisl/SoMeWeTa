#!/bin/bash

time ../bin/nlp4py-tagger --train tiger+decow+capit.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/decow14_capitalization_features.txt <(cut -f1,3 /data/Thomas/tiger_release_aug07.txt)
time ../bin/nlp4py-tagger --train tiger+decow+capit+empirist.model --prior tiger+decow+capit.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/decow14_capitalization_features.txt ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/train_all_sent_emo.txt
../bin/nlp4py-tagger --evaluate tiger+decow+capit+empirist.model ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/nlp4py-tagger --evaluate tiger+decow+capit+empirist.model ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/test_web_sent_emo.txt
