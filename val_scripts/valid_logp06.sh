#!/bin/bash

DIR=$1
ST=$2
ED=$3

for ((i=ST; i<=ED; i++)); do
    f=$DIR/model.$i
    if [ -e $f ]; then
        echo $f
        python decode.py --test ../data/molopt/logp06/valid.txt --vocab ../data/molopt/logp06/align_vocab.txt --model $f --latent_size 10 --hidden_size 270 --embed_size 200 --enum_root | python ../scripts/logp_score.py > $DIR/results.$i &
    fi
done
