#!/bin/bash

RUN=$1

echo "tiger" > tiger_results.$RUN.txt
time ../bin/somewe-tagger --train tiger.$RUN.model \
                          --brown /data/Thomas/decow14_brown.txt \
                          --lexicon /data/Thomas/morphy_medium+contractions.txt \
                          <(cut -f2,5 /data/Thomas/tiger_2.2_pav_instead_of_proav.conll09 \
                                      ../data/tiger_additional_quotes.conll09)
time ../bin/somewe-tagger --train tiger+empirist.$RUN.model \
                          --prior tiger.$RUN.model \
                          --brown /data/Thomas/decow14_brown.txt \
                          --lexicon /data/Thomas/morphy_medium+contractions.txt \
                          <(cat ../data/ignore/empirist_train.txt \
                                ../data/additional_training_german_web_social_media.txt)
../bin/somewe-tagger --evaluate tiger+empirist.$RUN.model ../data/ignore/empirist_test_cmc.txt >> tiger_results.$RUN.txt
../bin/somewe-tagger --evaluate tiger+empirist.$RUN.model ../data/ignore/empirist_test_web.txt >> tiger_results.$RUN.txt
