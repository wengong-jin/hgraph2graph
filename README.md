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
4. Make prediction on your lead compounds:
```
python ensemble_decode.py --test data/qed/valid.txt --vocab data/qed/vocab.txt --model_dir models/ > results.csv
```

The output is a CSV file having the following format:

| lead compound smiles | new compound smiles | similarity | 
| -------------------- | ------------------ | ---------- | 
| c1ccc(c2cncnc2)cc1[C@@]3(c4ccc(OC)cc4)N=C(N)OC3 | COc1ccc([C@@]2(c3cccc(C#N)c3)COC(N)=N2)cc1 | 0.6364 | 
| c1ccc(c2cncnc2)cc1[C@@]3(c4ccc(OC)cc4)N=C(N)OC3 | NC1=N[C@@](c2cccc(Cl)c2)(c2cccc(-c3ccccc3)c2)CO1 | 0.5273 | 
| c1ccc(c2cncnc2)cc1[C@@]3(c4ccc(OC)cc4)N=C(N)OC3 | CCOc1ccc([C@@]2([C@]3(c4ccc(OC)cc4)COC(N)=N3)COC(N)=N2)cc1 | 0.4310 |
| c1ccc(c2cncnc2)cc1[C@@]3(c4ccc(OC)cc4)N=C(N)OC3 | NC1=N[C@@](c2ccc(N)cc2)(c2ccc(-c3ccccc3)cc2)CO1 | 0.4717 |
| c1ccc(c2cncnc2)cc1[C@@]3(c4ccc(OC)cc4)N=C(N)OC3 | COc1ccc([C@@H](N)c2cccc(-c3cncnc3)c2)cc1 | 0.4643 | 
| c1ccc(c2cncnc2)cc1[C@@]3(c4ccc(OC)cc4)N=C(N)OC3 | NC1=N[C@@](c2cccc(N)c2)(c2cccc(-c3ccccc3)c2)CO1 | 0.5472 |

