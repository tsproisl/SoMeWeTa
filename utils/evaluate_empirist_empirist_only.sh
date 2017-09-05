#!/bin/bash

time ../bin/nlp4py-tagger --train empirist.model ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/train_all_sent_emo.txt
../bin/nlp4py-tagger --evaluate empirist.model ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/test_cmc_sent_emo.txt
../bin/nlp4py-tagger --evaluate empirist.model ~/Documents/Arbeit/nlp4py/data/empirist_gold_standard/test_web_sent_emo.txt
