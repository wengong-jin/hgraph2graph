#!/bin/bash

DIR=$1
ST=$2
ED=$3

mkdir $DIR/mode-1010
mkdir $DIR/mode-1001
mkdir $DIR/mode-0110

for ((i=ST; i<=ED; i++)); do
    f=$DIR/model.$i
    if [ -e $f ]; then
        echo $f
		python cond_decode.py --test ../data/molopt/multi-drd2-qed/valid.txt --vocab ../data/molopt/multi-drd2-qed/vocab.txt --hidden_size 270 --embed_size 200 --latent_size 4 --model $f --enum_root --cond 1,0,1,0 | python ../scripts/multi_score.py > $DIR/mode-1010/results.$i &
		python cond_decode.py --test ../data/molopt/multi-drd2-qed/valid.txt --vocab ../data/molopt/multi-drd2-qed/vocab.txt --hidden_size 270 --embed_size 200 --latent_size 4 --model $f --enum_root --cond 0,1,1,0 | python ../scripts/multi_score.py > $DIR/mode-0110/results.$i &
		python cond_decode.py --test ../data/molopt/multi-drd2-qed/valid.txt --vocab ../data/molopt/multi-drd2-qed/vocab.txt --hidden_size 270 --embed_size 200 --latent_size 4 --model $f --enum_root --cond 1,0,0,1 | python ../scripts/multi_score.py > $DIR/mode-1001/results.$i &
    fi
done
