# FALCON

Active-learning framework for drift-aware Android malware detection. A hierarchical contrastive encoder with a fixed buffer keeps per-round memory and retraining time near-constant while matching full-history retraining in new-family discovery.

## Usage
Download [this](https://drive.google.com/file/d/1O0upEcTolGyyvasCPkZFY86FNclk29XO/view) from Google Drive. The zipped file contains DREBIN features of the APIGraph dataset and AndroZoo dataset we used in the paper.

Extract the downloaded file to data/, such that the datasets are under data/gen_apigraph_drebin and data/gen_androzoo_drebin.

Download the environment.yml file and create the Conda environment using Anaconda:
conda env create -f environment.yml
conda activate <environment_name>
Run this command to use our FALCON
```bash
SEQ=088
LR=0.003
OPT=sgd
SCH=step
DECAY=0.95
E=250
WLR=0.00015
WE=100
DATA=gen_apigraph_drebin
TRAIN_START=2012-01
TRAIN_END=2012-12
TEST_START=2013-01
TEST_END=2018-12
RESULT_DIR=testing
AL_OPT=adam

CNT=400
INDEX=$((CNT / 2))
SEED=42


modeldim="512-384-256-128"
S='half'
B=1024
LOSS='hi-dist-xent'
TS=$(date "+%m.%d-%H.%M.%S")
python -u relabel_falcon.py	                                \
            --data ${DATA}                                  \
            --benign_zero                                   \
            --mdate 20230501                                \
            --train_start ${TRAIN_START}                    \
            --train_end ${TRAIN_END}                        \
            --test_start ${TEST_START}                      \
            --test_end ${TEST_END}                          \
            --encoder simple-enc-mlp                        \
            --classifier simple-enc-mlp                     \
            --loss_func ${LOSS}                             \
            --enc-hidden ${modeldim}                        \
            --mlp-hidden 100-100                            \
            --mlp-dropout 0.2                               \
            --sampler ${S}                                  \
            --bsize ${B}                                    \
            --optimizer ${OPT}                              \
            --scheduler ${SCH}                              \
            --learning_rate ${LR}                           \
            --lr_decay_rate ${DECAY}                        \
            --lr_decay_epochs "10,500,10"                   \
            --epochs ${E}                                   \
            --encoder-retrain                               \
            --al_optimizer ${AL_OPT}                        \
            --warm_learning_rate ${WLR}                     \
            --al_epochs ${WE}                               \
            --xent-lambda 100                               \
            --display-interval 180                          \
            --al                                            \
            --count ${CNT}                                  \
            --local_pseudo_loss                             \
            --reduce "none"                                 \
            --sample_reduce 'mean'                          \
            --seed ${SEED}                                  \
            --index ${INDEX}                                \
            --result experiments/cnt${CNT}_525mal_${INDEX}unc_${INDEX}c.csv \
            --log_path experiments/cnt${CNT}_525mal_${INDEX}unc_${INDEX}c.log \
            >> experiments/cnt${CNT}_525mal_${INDEX}unc_${INDEX}c.log 2>&1 &

```

`--count`: monthly labeling budget · `--local_pseudo_loss`: FALCON's selector · seeds: 42, 123, 2026.

**Outputs:** training log (buffer memory, training time), per-round results CSV
