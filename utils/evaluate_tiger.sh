#!/bin/bash

RUN=$1

echo "tiger" > tiger_results.$RUN.txt
time ../bin/somewe-tagger --train tiger.$RUN.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/morphy_wc.txt <(cut -f 2,5 /data/Thomas/tiger_2.2_pav_instead_of_proav.conll09 ../data/tiger_additional_quotes.conll09)
time ../bin/somewe-tagger --train tiger+empirist.$RUN.model --prior tiger.$RUN.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/morphy_wc.txt <(cat ../data/ignore/empirist_gold_standard/train_all_sent_emo.txt ../data/additional_training_german_web_social_media.txt)
../bin/somewe-tagger --evaluate tiger+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt >> tiger_results.$RUN.txt
../bin/somewe-tagger --evaluate tiger+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt >> tiger_results.$RUN.txt
