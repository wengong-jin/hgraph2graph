# Atom-by-Atom Translation Baseline

Our paper is at https://arxiv.org/abs/1907.11223

1. Train the model:
```
mkdir models/
python gnn_train.py --train ../data/qed/train_pairs.txt --save_dir models/ 
```
2. Make prediction on your lead compounds:
```
python decode.py --test data/qed/valid.txt --model models/model.5 > results.txt
```

