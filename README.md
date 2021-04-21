# Hierarchical Generation of Molecular Graphs using Structural Motifs

Our paper is at https://arxiv.org/pdf/2002.03230.pdf

## Installation
First install the dependencies via conda:
 * PyTorch >= 1.0.0
 * networkx
 * RDKit >= 2019.03
 * numpy
 * Python >= 3.6

And then run `pip install .`

## Data Format
* For graph generation, each line of a training file is a SMILES string of a molecule
* For graph translation, each line of a training file is a pair of molecules (molA, molB) that are similar to each other but molB has better chemical properties. Please see `data/qed/train_pairs.txt`. The test file is a list of molecules to be optimized. Please see `data/qed/test.txt`.

## Graph generation training procedure
1. Extract substructure vocabulary from a given set of molecules:
```
python get_vocab.py --ncpu 16 < data/qed/mols.txt > vocab.txt
```

2. Preprocess training data:
```
python preprocess.py --train data/qed/mols.txt --vocab data/qed/vocab.txt --ncpu 16 --mode single
mkdir train_processed
mv tensor* train_processed/
```

3. Train graph generation model
```
mkdir ckpt/generation
python train_generator.py --train train_processed/ --vocab data/qed/vocab.txt --save_dir ckpt/generation
```

4. Sample molecules from a model checkpoint
```
python generate.py --vocab data/qed/vocab.txt --model ckpt/generation/model.5 --nsamples 1000
```

## Graph translation training procedure
1. Extract substructure vocabulary from a given set of molecules:
```
python get_vocab.py --ncpu 16 < data/qed/mols.txt > vocab.txt
```
Please replace `data/qed/mols.txt` with your molecules.

2. Preprocess training data:
```
python preprocess.py --train data/qed/train_pairs.txt --vocab data/qed/vocab.txt --ncpu 16
mkdir train_processed
mv tensor* train_processed/
```

3. Train the model:
```
mkdir ckpt/translation
python train_translator.py --train train_processed/ --vocab data/qed/vocab.txt --save_dir ckpt/translation
```

4. Make prediction on your lead compounds (you can use any model checkpoint, here we use model.5 for illustration)
```
python translate.py --test data/qed/valid.txt --vocab data/qed/vocab.txt --model ckpt/translation/model.5 --num_decode 20 > results.csv
```

## Polymer generation
The polymer generation code is in the `polymer/` folder. The polymer generation code is similar to `train_generator.py`, but the substructures are tailored for polymers. 
For generating regular drug like molecules, we recommend to use `train_generator.py` in the root directory.

