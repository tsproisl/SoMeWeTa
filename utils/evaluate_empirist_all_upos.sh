#!/bin/bash

RUN=$1

echo "tiger+decow+morphy+empirist"
time ../bin/somewe-tagger --train upos_tiger+decow+morphy.$RUN.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/morphy_wc.txt /data/Thomas/tiger_ud_upos.txt
time ../bin/somewe-tagger --train upos_tiger+decow+morphy+empirist.$RUN.model --prior upos_tiger+decow+morphy.$RUN.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/morphy_wc.txt ../data/ignore/train_all_sent_emo_upos.txt
../bin/somewe-tagger --evaluate upos_tiger+decow+morphy+empirist.$RUN.model ../data/ignore/test_cmc_sent_emo_upos.txt
../bin/somewe-tagger --evaluate upos_tiger+decow+morphy+empirist.$RUN.model ../data/ignore/test_web_sent_emo_upos.txt

echo "tiger+decow+morphy+empirist+extra"
time ../bin/somewe-tagger --train upos_tiger+decow+morphy+empirist+extra.$RUN.model --prior upos_tiger+decow+morphy.$RUN.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/morphy_wc.txt <(cat ../data/ignore/train_all_sent_emo_upos.txt ../data/authentic_additional_training_german_web_social_media_upos.txt)
../bin/somewe-tagger --evaluate upos_tiger+decow+morphy+empirist+extra.$RUN.model ../data/ignore/test_cmc_sent_emo_upos.txt
../bin/somewe-tagger --evaluate upos_tiger+decow+morphy+empirist+extra.$RUN.model ../data/ignore/test_web_sent_emo_upos.txt
