#!/bin/bash

FILE="$1"
BIN=../bin
FOLDS=10

# reformat as one sentence per line
tr '\n' ' ' < "$FILE" | sed -re 's/ {2,}/\n/g' > tmp_cv_osl

# optional: shuffle the file
# shuf tmp_cv_osl > tmp_cv_osl_shuf
# mv tmp_cv_osl_shuf tmp_cv_osl

# split into $FOLDS parts
split --number=l/$FOLDS --numeric-suffixes --suffix-length=1 tmp_cv_osl tmp_cv_i_

# reformat as one token per line
for (( i=0; i < $FOLDS; i++ ))
do
    sed -re 's/$/\n/' tmp_cv_i_$i | tr ' ' '\n' > tmp_cv_tmp
    mv tmp_cv_tmp tmp_cv_i_$i
done

# train, tag and evaluate
ACCURACIES=()
for (( i=0; i < $FOLDS; i++ ))
do
    echo "fold no. $i"
    echo "-----------"
    mv tmp_cv_i_$i tmp_cv_test
    cat tmp_cv_i_* > tmp_cv_train
    cat tmp_cv_train | "${BIN}/somewe-tagger" --train tmp_cv_model --prior ../data/german_newspaper.model --brown /data/Thomas/decow14_brown.txt --lexicon /data/Thomas/morphy_wc.txt -
    if [ "$?" -ne "0" ]; then
	echo "Training failed"
	exit 1
    fi
    cut -f1 tmp_cv_test | "${BIN}/somewe-tagger" --tag tmp_cv_model - > tmp_cv_output
    paste tmp_cv_test <(cut -f2 tmp_cv_output) > tmp_cv_eval
    # evaluate
    ACCURACIES+=($(awk '! /^\s*$/ {total += 1; if ($2 == $3) {correct += 1}} END {print correct/total*100}' tmp_cv_eval))
    rm tmp_cv_model
    mv tmp_cv_test tmp_cv_i_$i
    echo ""
done

# calculate the mean accuracy
SUM=0
for a in ${ACCURACIES[*]}
do
    SUM=$(echo "${SUM} + ${a}" | bc -l)
done
MEAN=$(echo "${SUM} / ${FOLDS}" | bc -l)

# calculate the standard deviation
SUM=0
for a in ${ACCURACIES[*]}
do
    SUM=$(echo "${SUM} + (${a} - ${MEAN}) ^ 2" | bc -l)
done
STD=$(echo "sqrt( ${SUM} / ${FOLDS} )" | bc -l)
DOUBLE_STD=$(echo "2 * ${STD}" | bc -l)

# print evaluation results
echo "RESULTS"
echo "-------"
# echo "mean: $MEAN"
# echo "std: $STD"
printf "Accuracy: %.2f (±%.2f)\n" $(echo $MEAN | tr '.' ',') $(echo $DOUBLE_STD | tr '.' ',')

# clean up
rm tmp_cv_*
