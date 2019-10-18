#!/bin/bash

DIR=$1
ST=$2
ED=$3

for ((i=ST; i<=ED; i++)); do
    f=$DIR/model.$i
    if [ -e $f ]; then
        echo $f
        python decode.py --test data/logp04/valid.txt --vocab data/logp04/vocab.txt --model $f --hidden_size 170 --embed_size 170 | python scripts/logp_score.py  > $DIR/results.$i &
    fi
done
