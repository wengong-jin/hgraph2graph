# Hierarchical Generation of Molecular Graphs using Structural Motifs

Our paper is at https://arxiv.org/pdf/2002.03230.pdf

## Installation
First install the dependencies via conda:
 * PyTorch >= 1.0.0
 * networkx
 * RDKit
 * numpy
 * Python >= 3.6

And then run `pip install .`

## Molecule Generation
The molecule generation code is in the `generation/` folder.

## Graph translation Data Format
* The training file should contain pairs of molecules (molA, molB) that are similar to each other but molB has better chemical properties. Please see `data/qed/train_pairs.txt`.
* The test file is a list of molecules to be optimized. Please see `data/qed/test.txt`.

## Graph translation training procedure
1. Extract substructure vocabulary from a given set of molecules:
```
python get_vocab.py < data/qed/mols.txt > vocab.txt
```
Please replace `data/qed/mols.txt` with your molecules data file.

2. Preprocess training data:
```
python preprocess.py --train data/qed/train_pairs.txt --vocab data/qed/vocab.txt --ncpu 16 < data/qed/train_pairs.txt
mkdir train_processed
mv tensor* train_processed/
```
Please replace `--train` and `--vocab` with training and vocab file.

3. Train the model:
```
mkdir models/
python gnn_train.py --train train_processed/ --vocab data/qed/vocab.txt --save_dir models/ 
```

4. Make prediction on your lead compounds (you can use any model checkpoint, here we use model.5 for illustration)
```
python decode.py --test data/qed/valid.txt --vocab data/qed/vocab.txt --model models/model.5 --num_decode 20 > results.csv
```
