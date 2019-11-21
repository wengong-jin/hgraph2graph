# Atom-by-Atom Translation Baseline

Description of atom-by-atom translation baseline is in the appendix of our paper (https://arxiv.org/abs/1907.11223)

1. Train the model:
```
mkdir models/
python gnn_train.py --train ../data/qed/train_pairs.txt --save_dir models/ 
```
2. Make prediction on your lead compounds:
```
python decode.py --test data/qed/valid.txt --model models/model.5 > results.txt
```

