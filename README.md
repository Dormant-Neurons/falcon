# FALCON

Active-learning framework for drift-aware Android malware detection. A hierarchical contrastive encoder with a fixed buffer keeps per-round memory and retraining time near-constant while matching full-history retraining in new-family discovery.

**Requirements:** Python 3.9+, PyTorch, NumPy, SciPy, scikit-learn, pandas.

**Dataset:** APIGraph/Drebin (`gen_apigraph_drebin`); train 2012, monthly retraining 2013–2018.

## Usage

```bash
python -u relabel_falcon.py \
    --data gen_apigraph_drebin
    --benign_zero --mdate 20230501 \
    --train_start 2012-01 --train_end 2012-12 \
    --test_start 2013-01 --test_end 2018-12 \
    --encoder simple-enc-mlp --classifier simple-enc-mlp \
    --loss_func hi-dist-xent --enc-hidden 512-384-256-128 \
    --mlp-hidden 100-100 --mlp-dropout 0.2 --sampler half --bsize 1024 \
    --optimizer sgd --scheduler step --learning_rate 0.003 \
    --lr_decay_rate 0.95 --lr_decay_epochs "10,500,10" \
    --epochs 250 --encoder-retrain \
    --al --al_optimizer adam --al_epochs 100 --warm_learning_rate 0.00015 \
    --xent-lambda 100 --count 800 --index 400 \
    --local_pseudo_loss --reduce "none" --sample_reduce mean --seed 42 \
    --result <output>.csv --log_path <output>.log
```

`--count`: monthly labeling budget · `--local_pseudo_loss`: FALCON's selector · seeds: 42, 123, 2026.

**Outputs:** training log (buffer memory, training time), per-round results CSV
