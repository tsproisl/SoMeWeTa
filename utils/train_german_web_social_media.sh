#!/bin/bash

# prior
../bin/somewe-tagger --train ../data/ignore/german_newspaper_for_empirist_yyyy-mm-dd.model \
		     --brown /data/Thomas/decow14_brown.txt \
		     --lexicon /data/Thomas/morphy_medium+contractions.txt \
		     <(cut -f2,5 /data/Thomas/tiger_2.2_pav_instead_of_proav.conll09 \
			         ../data/tiger_additional_quotes.conll09)

# cmc model
../bin/somewe-tagger --train ../data/ignore/german_web_social_media_yyyy-mm-dd.model \
		     --prior ../data/ignore/german_newspaper_for_empirist_yyyy-mm-dd.model \
		     --brown /data/Thomas/decow14_brown.txt \
		     --lexicon /data/Thomas/morphy_medium+contractions.txt \
		     <(cat ../data/ignore/empirist_train.txt \
			   ../data/ignore/empirist_test.txt \
			   ../data/additional_training_german_web_social_media.txt)
