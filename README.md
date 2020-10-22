# Hierarchical Graph-to-Graph Translation for Molecules

Our paper is at https://arxiv.org/abs/1907.11223

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
python preprocess.py --train data/qed/train_pairs.txt --vocab data/qed/vocab.txt --ncpu 16
mkdir train_processed
mv tensor* train_processed/
```
Please replace `--train` and `--vocab` with training and vocab file.

3. Train the model:
```
mkdir models/
python gnn_train.py --train train_processed/ --vocab data/qed/vocab.txt --save_dir models/ 
```

4. Make prediction on your lead compounds:
```
python decode.py --test data/qed/valid.txt --vocab data/qed/vocab.txt --model models/model.5 --num_decode 20 > results.csv
```

## Sample training procedure for conditional generation
For conditional graph translation (QED + DRD2), you can first preprocess the data by:
```
python preprocess.py --train data/multi-qed-drd2/train_pairs.txt --vocab data/multi-qed-drd2/vocab.txt --ncpu 16  --mode cond_pair
mkdir train_processed
mv tensor* train_processed/
```
To train a model, run:
```
python gnn_train.py --train train_processed/ --vocab data/multi-qed-drd2/vocab.txt --save_dir models/ --conditional
```
Finally, translate test compounds by
```
python cond_decode.py --test data/multi-qed-drd2/valid.txt --vocab data/multi-qed-drd2/vocab.txt --cond 1,0,1,0 --model models/model.5 --num_decode 20 > results.csv
```
where `1,0,1,0` means high QED, high DRD2. Change it to `1,0,0,1` for high QED, low DRD2 and `0,1,1,0` for low QED, high DRD2.
