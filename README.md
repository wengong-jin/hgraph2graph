# Hierarchical Generation of Molecular Graphs using Structural Motifs

Our paper is at https://arxiv.org/pdf/2002.03230.pdf

## Installation
First install the dependencies via conda:
 * PyTorch >= 1.0.0
 * networkx
 * RDKit >= 2019.03
 * numpy
 * Python >= 3.6

And then run `pip install .`. Additional dependency for property-guided finetuning:
 * Chemprop >= 1.2.0


## Data Format
* For graph generation, each line of a training file is a SMILES string of a molecule
* For graph translation, each line of a training file is a pair of molecules (molA, molB) that are similar to each other but molB has better chemical properties. Please see `data/qed/train_pairs.txt`. The test file is a list of molecules to be optimized. Please see `data/qed/test.txt`.

## Molecule generation pretraining procedure
We can train a molecular language model on a large corpus of unlabeled molecules. We have uploaded a model checkpoint pre-trained on ChEMBL dataset in `ckpt/chembl-pretrained/model.ckpt`. If you wish to train your own language model, please follow the steps below:

1. Extract substructure vocabulary from a given set of molecules:
```
python get_vocab.py --ncpu 16 < data/chembl/all.txt > vocab.txt
```

2. Preprocess training data:
```
python preprocess.py --train data/chembl/all.txt --vocab data/chembl/all.txt --ncpu 16 --mode single
mkdir train_processed
mv tensor* train_processed/
```

3. Train graph generation model
```
mkdir ckpt/chembl-pretrained
python train_generator.py --train train_processed/ --vocab data/chembl/vocab.txt --save_dir ckpt/chembl-pretrained
```

4. Sample molecules from a model checkpoint
```
python generate.py --vocab data/chembl/vocab.txt --model ckpt/chembl-pretrained/model.ckpt --nsamples 1000
```

## Property-guided molcule generation procedure (a.k.a. finetuning)
The following script loads a trained Chemprop model and finetunes a pre-trained molecule language model to generate molecules with specific chemical properties.
```
mkdir ckpt/finetune
python finetune_generator.py --train ${ACTIVE_MOLECULES} --vocab data/chembl/vocab.txt --generative_model ckpt/chembl-pretrained/model.ckpt --chemprop_model ${YOUR_PROPERTY_PREDICTOR} --min_similarity 0.1 --max_similarity 0.5 --nsample 10000 --epoch 10 --threshold 0.5 --save_dir ckpt/finetune
```
Here `${ACTIVE_MOLECULES}` should contain a list of experimentally verified active molecules. 

`${YOUR_PROPERTY_PREDICTOR}` should be a directory containing saved chemprop model checkpoint. 

`--max_similarity 0.5` means any novel molecule should have nearest neighbor similarity lower than 0.5 to any known active molecules in ${ACTIVE_MOLECULES}` file.

`--nsample 10000` means to sample 10000 molecules in each epoch. 

`--threshold 0.5` is the activity threshold. A molecule is considered as active if its predicted chemprop score is greater than 0.5.

In each epoch, generated active molecules are saved in `ckpt/finetune/good_molecules.${epoch}`. All the novel active molecules are saved in `ckpt/finetune/new_molecules.${epoch}`

## Molecule translation training procedure
Molecule translation is often useful for lead optimization (i.e., modifying a given molecule to improve its properties)

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

