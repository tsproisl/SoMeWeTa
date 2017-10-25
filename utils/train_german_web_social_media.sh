#!/bin/bash

../bin/somewe-tagger --train ../data/ignore/german_web_social_media_vX.Y.Z.model --prior ../data/ignore/german_newspaper_v1.0.0.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/morphy_wc.txt <(cat ../data/ignore/empirist_gold_standard/train_all_sent_emo.txt ../data/ignore/empirist_gold_standard/test_all_sent_emo.txt ../data/additional_training_german_web_social_media.txt)
