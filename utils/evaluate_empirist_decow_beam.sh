#!/bin/bash

time ../bin/nlp4py-tagger -i 12 -b 6 --train tiger+decow_beam.model --brown /data/decow14_brown.txt <(cut -f1,3 /cip/corpora/German/Tiger/Tiger2.1/corpus/tiger_release_aug07.txt)
time ../bin/nlp4py-tagger -i 12 -b 6 --train tiger+decow+empirist_beam.model --prior tiger+decow_beam.model --brown /data/decow14_brown.txt ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/train_all_sent_emo.txt
../bin/nlp4py-tagger -i 12 -b 6 --evaluate tiger+decow+empirist_beam.model ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/nlp4py-tagger -i 12 -b 6 --evaluate tiger+decow+empirist_beam.model ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/test_web_sent_emo.txt
