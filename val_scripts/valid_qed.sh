#!/bin/bash

DIR=$1
ST=$2
ED=$3

for ((i=ST; i<=ED; i++)); do
    f=$DIR/model.$i
    if [ -e $f ]; then
        echo $f
        python decode.py --test ../data/molopt/qed/valid.txt --vocab ../data/molopt/qed/align_vocab.txt --model $f --enum_root --hidden_size 270 --embed_size 200 | python ../scripts/qed_score.py > $DIR/results.$i &
    fi
done
