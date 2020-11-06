# creating a wmt_en_ro largish subset for tests 

This is to support an examples/seq2seq/finetune.py test which needed to run:
```
finetune.py --n_train 100000 --n_val 500 --n_test 500 [...]
```

This subset was derived from the full wmt_en_ro dataset with:
```
wget https://cdn-datasets.huggingface.co/translation/wmt_en_ro.tar.gz
tar -xvzf wmt_en_ro.tar.gz

mkdir wmt_en_ro-tr100k-va0.5k-te0.5k
head -100000 wmt_en_ro/test.source  > wmt_en_ro-tr100k-va0.5k-te0.5k/test.source
head -100000 wmt_en_ro/test.target  > wmt_en_ro-tr100k-va0.5k-te0.5k/test.target
head -500    wmt_en_ro/train.source > wmt_en_ro-tr100k-va0.5k-te0.5k/train.source
head -500    wmt_en_ro/train.target > wmt_en_ro-tr100k-va0.5k-te0.5k/train.target
head -500    wmt_en_ro/val.source   > wmt_en_ro-tr100k-va0.5k-te0.5k/val.source
head -500    wmt_en_ro/val.target   > wmt_en_ro-tr100k-va0.5k-te0.5k/val.target
tar -cvzf wmt_en_ro-tr100k-va0.5k-te0.5k.tar.gz wmt_en_ro-tr100k-va0.5k-te0.5k

export ds=s3://datasets.huggingface.co
aws s3 cp wmt_en_ro-tr100k-va0.5k-te0.5k.tar.gz $ds/translation/
```

now we can get it via:

https://cdn-datasets.huggingface.co/translation/wmt_en_ro-tr100k-va0.5k-te0.5k.tar.gz

The original was 56M, this one is just 256K.
