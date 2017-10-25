#!/bin/bash

RUN=$1

../data/ignore/empirist_gold_standard/

echo "empirist"
time ../bin/somewe-tagger --train empirist.$RUN.model ../data/ignore/empirist_gold_standard/train_all_sent_emo.txt
../bin/somewe-tagger --evaluate empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/somewe-tagger --evaluate empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt

echo "tiger"
time ../bin/somewe-tagger --train tiger.$RUN.model <(cut -f1,3 /data/Thomas/tiger_release_aug07.txt)
../bin/somewe-tagger --evaluate tiger.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/somewe-tagger --evaluate tiger.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt

echo "tiger+empirist"
time ../bin/somewe-tagger --train tiger+empirist.$RUN.model --prior tiger.$RUN.model ../data/ignore/empirist_gold_standard/train_all_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt

echo "tiger+morphy+empirist"
time ../bin/somewe-tagger --train tiger+morphy.$RUN.model --lexicon /data/Thomas/morphy_wc.txt <(cut -f1,3 /data/Thomas/tiger_release_aug07.txt)
time ../bin/somewe-tagger --train tiger+morphy+empirist.$RUN.model --prior tiger+morphy.$RUN.model --lexicon /data/Thomas/morphy_wc.txt ../data/ignore/empirist_gold_standard/train_all_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+morphy+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+morphy+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt

# echo "tiger+dewikiall+empirist"
# time ../bin/somewe-tagger --train tiger+dewikiall.$RUN.model --brown /data/Thomas/dewiki_all_brown.txt <(cut -f1,3 /data/Thomas/tiger_release_aug07.txt)
# time ../bin/somewe-tagger --train tiger+dewikiall+empirist.$RUN.model --prior tiger+dewikiall.$RUN.model --brown /data/Thomas/dewiki_all_brown.txt ../data/ignore/empirist_gold_standard/train_all_sent_emo.txt
# ../bin/somewe-tagger --evaluate tiger+dewikiall+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
# ../bin/somewe-tagger --evaluate tiger+dewikiall+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt

# echo "tiger+dewikiall+capit+empirist"
# time ../bin/somewe-tagger --train tiger+dewikiall+capit.$RUN.model --brown /data/Thomas/dewiki_all_brown.txt --lexicon /data/Thomas/dewiki-20170620_all_capitalization_features.txt <(cut -f1,3 /data/Thomas/tiger_release_aug07.txt)
# time ../bin/somewe-tagger --train tiger+dewikiall+capit+empirist.$RUN.model --prior tiger+dewikiall+capit.$RUN.model --brown /data/Thomas/dewiki_all_brown.txt --lexicon /data/Thomas/dewiki-20170620_all_capitalization_features.txt ../data/ignore/empirist_gold_standard/train_all_sent_emo.txt
# ../bin/somewe-tagger --evaluate tiger+dewikiall+capit+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
# ../bin/somewe-tagger --evaluate tiger+dewikiall+capit+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt

# echo "tiger+dewikiall+morphy+empirist"
# time ../bin/somewe-tagger --train tiger+dewikiall+morphy.$RUN.model --brown /data/Thomas/dewiki_all_brown.txt --lexicon /data/Thomas/morphy_wc.txt <(cut -f1,3 /data/Thomas/tiger_release_aug07.txt)
# time ../bin/somewe-tagger --train tiger+dewikiall+morphy+empirist.$RUN.model --prior tiger+dewikiall+morphy.$RUN.model --brown /data/Thomas/dewiki_all_brown.txt --lexicon /data/Thomas/morphy_wc.txt ../data/ignore/empirist_gold_standard/train_all_sent_emo.txt
# ../bin/somewe-tagger --evaluate tiger+dewikiall+morphy+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
# ../bin/somewe-tagger --evaluate tiger+dewikiall+morphy+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt

# echo "tiger+dewikiall+morcap+empirist"
# time ../bin/somewe-tagger --train tiger+dewikiall+morcap.$RUN.model --brown /data/Thomas/dewiki_all_brown.txt --lexicon morphy+dewikiall_capitalization.txt <(cut -f1,3 /data/Thomas/tiger_release_aug07.txt)
# time ../bin/somewe-tagger --train tiger+dewikiall+morcap+empirist.$RUN.model --prior tiger+dewikiall+morcap.$RUN.model --brown /data/Thomas/dewiki_all_brown.txt --lexicon /data/Thomas/morphy+dewikiall_capitalization.txt ../data/ignore/empirist_gold_standard/train_all_sent_emo.txt
# ../bin/somewe-tagger --evaluate tiger+dewikiall+morcap+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
# ../bin/somewe-tagger --evaluate tiger+dewikiall+morcap+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt

echo "tiger+decow+empirist"
time ../bin/somewe-tagger --train tiger+decow.$RUN.model --brown /data/Thomas/decow14_brown.txt <(cut -f1,3 /data/Thomas/tiger_release_aug07.txt)
time ../bin/somewe-tagger --train tiger+decow+empirist.$RUN.model --prior tiger+decow.$RUN.model --brown /data/Thomas/decow14_brown.txt ../data/ignore/empirist_gold_standard/train_all_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+decow+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+decow+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt

echo "tiger+decow+capit+empirist"
time ../bin/somewe-tagger --train tiger+decow+capit.$RUN.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/decow14_capitalization_features.txt <(cut -f1,3 /data/Thomas/tiger_release_aug07.txt)
time ../bin/somewe-tagger --train tiger+decow+capit+empirist.$RUN.model --prior tiger+decow+capit.$RUN.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/decow14_capitalization_features.txt ../data/ignore/empirist_gold_standard/train_all_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+decow+capit+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+decow+capit+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt

echo "tiger+decow+morphy+empirist"
time ../bin/somewe-tagger --train tiger+decow+morphy.$RUN.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/morphy_wc.txt <(cut -f1,3 /data/Thomas/tiger_release_aug07.txt)
time ../bin/somewe-tagger --train tiger+decow+morphy+empirist.$RUN.model --prior tiger+decow+morphy.$RUN.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/morphy_wc.txt ../data/ignore/empirist_gold_standard/train_all_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+decow+morphy+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+decow+morphy+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt

echo "tiger+decow+morcap+empirist"
time ../bin/somewe-tagger --train tiger+decow+morcap.$RUN.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/morphy+decow14_capitalization.txt <(cut -f1,3 /data/Thomas/tiger_release_aug07.txt)
time ../bin/somewe-tagger --train tiger+decow+morcap+empirist.$RUN.model --prior tiger+decow+morcap.$RUN.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/morphy+decow14_capitalization.txt ../data/ignore/empirist_gold_standard/train_all_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+decow+morcap+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+decow+morcap+empirist.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt

echo "tiger+decow+morphy+empirist+extra"
time ../bin/somewe-tagger --train tiger+decow+morphy+empirist+extra.$RUN.model --prior tiger+decow+morphy.$RUN.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/morphy_wc.txt <(cat ../data/ignore/empirist_gold_standard/train_all_sent_emo.txt ../data/additional_training_german_web_social_media.txt)
../bin/somewe-tagger --evaluate tiger+decow+morphy+empirist+extra.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+decow+morphy+empirist+extra.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt

echo "tiger+decow+morcap+empirist+extra"
time ../bin/somewe-tagger --train tiger+decow+morcap+empirist+extra.$RUN.model --prior tiger+decow+morcap.$RUN.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/morphy+decow14_capitalization.txt <(cat ../data/ignore/empirist_gold_standard/train_all_sent_emo.txt ../data/additional_training_german_web_social_media.txt)
../bin/somewe-tagger --evaluate tiger+decow+morcap+empirist+extra.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+decow+morcap+empirist+extra.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt

echo "tiger+decow+morcap+cmc"
time ../bin/somewe-tagger --train tiger+decow+morcap+cmc.$RUN.model --prior tiger+decow+morcap.$RUN.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/morphy+decow14_capitalization.txt ../data/ignore/empirist_gold_standard/train_cmc_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+decow+morcap+cmc.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+decow+morcap+cmc.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt

echo "tiger+decow+morcap+web"
time ../bin/somewe-tagger --train tiger+decow+morcap+web.$RUN.model --prior tiger+decow+morcap.$RUN.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/morphy+decow14_capitalization.txt ../data/ignore/empirist_gold_standard/train_web_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+decow+morcap+web.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+decow+morcap+web.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt

echo "tiger+decow+morcap+cmc+extra"
time ../bin/somewe-tagger --train tiger+decow+morcap+cmc+extra.$RUN.model --prior tiger+decow+morcap.$RUN.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/morphy+decow14_capitalization.txt <(cat ../data/ignore/empirist_gold_standard/train_cmc_sent_emo.txt ../data/additional_training_german_web_social_media.txt)
../bin/somewe-tagger --evaluate tiger+decow+morcap+cmc+extra.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+decow+morcap+cmc+extra.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt

echo "tiger+decow+morcap+web+extra"
time ../bin/somewe-tagger --train tiger+decow+morcap+web+extra.$RUN.model --prior tiger+decow+morcap.$RUN.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/morphy+decow14_capitalization.txt <(cat ../data/ignore/empirist_gold_standard/train_web_sent_emo.txt ../data/additional_training_german_web_social_media.txt)
../bin/somewe-tagger --evaluate tiger+decow+morcap+web+extra.$RUN.model ../data/ignore/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/somewe-tagger --evaluate tiger+decow+morcap+web+extra.$RUN.model ../data/ignore/empirist_gold_standard/test_web_sent_emo.txt
