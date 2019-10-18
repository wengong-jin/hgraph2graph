# Hierarchical Graph-to-Graph Translation for Molecules

## Installation
First install the dependencies via conda:
 * PyTorch >= 1.0.0
 * networkx
 * RDKit
 * numpy
 * Python >= 3.6

And then run `pip install .`

## Data Format
* The training file should contain pairs of molecules (molA, molB) that are similar to each other but molB has better chemical properties. Please see `data/qed/train_pairs.txt`.
* The test file is a list of molecules to be optimized. Please see `data/qed/test.txt`.

## Sample training procedure
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
