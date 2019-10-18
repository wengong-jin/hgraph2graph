#!/bin/bash

DIR=$1
ST=$2
ED=$3

for ((i=ST; i<=ED; i++)); do
    f=$DIR/model.$i
    if [ -e $f ]; then
        echo $f
        python decode.py --test ../data/molopt/drd2/valid.txt --vocab ../data/molopt/drd2/align_vocab.txt --enum_root --model $f --hidden_size 270 --embed_size 200 | python ../scripts/drd2_score.py > $DIR/results.$i &
    fi
done
