#!/bin/bash

time ../bin/nlp4py-tagger --train tiger+decow+morphy+mapping.model --brown /data/Thomas/decow14_brown.txt --lexicon /cip/corpora/Lexicon/Morphy/morphy_wc.txt --mapping ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/stts2_to_upos.json <(cut -f1,3 /data/Thomas/tiger_release_aug07.txt)
time ../bin/nlp4py-tagger --train tiger+decow+morphy+mapping+empirist.model --prior tiger+decow+morphy+mapping.model --brown /data/Thomas/decow14_brown.txt --lexicon /cip/corpora/Lexicon/Morphy/morphy_wc.txt  --mapping ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/stts2_to_upos.json ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/train_all_sent_emo.txt
../bin/nlp4py-tagger --evaluate tiger+decow+morphy+mapping+empirist.model ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/nlp4py-tagger --evaluate tiger+decow+morphy+mapping+empirist.model ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/test_web_sent_emo.txt
