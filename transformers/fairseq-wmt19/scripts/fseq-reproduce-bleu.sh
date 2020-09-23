#/bin/bash

# this script reproduces fairseq wmt19 BLEU scores
# see: https://github.com/pytorch/fairseq/issues/2544

# en-ru

git clone https://github.com/pytorch/fairseq/
cd fairseq

git clone https://github.com/moses-smt/mosesdecoder
git clone git@github.com:glample/fastBPE.git
cd fastBPE; g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast; cd -

mkdir -p data-bin

#get model
curl --output - https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz | tar xvzf - -C data_bin

export PAIR=en-ru
export DATA_DIR=data-bin/wmt19.en-ru.ensemble

# get evaluation data
sacrebleu -t wmt19 -l $PAIR --echo src > $DATA_DIR/test.en-ru.en
sacrebleu -t wmt19 -l $PAIR --echo ref > $DATA_DIR/test.en-ru.ru

# normalize and tokenize
cat $DATA_DIR/test.en-ru.en | ./mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en | ./mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l en -q > $DATA_DIR/temp.en-ru.en
cat $DATA_DIR/test.en-ru.ru | ./mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l ru | ./mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l ru -q > $DATA_DIR/temp.en-ru.ru

# download BPE codes (temporary, wmt19.en-ru.ensemble.tar.gz is waiting to be fixed to include the right codes), otherwise will use those in DATA_DIR
wget -P $DATA_DIR https://dl.fbaipublicfiles.com/fairseq/ru24k.fastbpe.code
wget -P $DATA_DIR https://dl.fbaipublicfiles.com/fairseq/en24k.fastbpe.code
# apply BPE
./fastBPE/fast applybpe $DATA_DIR/test.en-ru.en $DATA_DIR/temp.en-ru.en $DATA_DIR/en24k.fastbpe.code
./fastBPE/fast applybpe $DATA_DIR/test.en-ru.ru $DATA_DIR/temp.en-ru.ru $DATA_DIR/ru24k.fastbpe.code

# which checkpoints to eval against (all or just a specific one)
#export CHKPT=$DATA_DIR/model1.pt:$DATA_DIR/model2.pt:$DATA_DIR/model3.pt:$DATA_DIR/model4.pt
export CHKPT=$DATA_DIR/model4.pt

# generate (w/ 4 model ensemble)
fairseq-generate $DATA_DIR --path $CHKPT \
--beam 5 --batch-size 32 --remove-bpe --source-lang en --target-lang ru --task translation --dataset-impl raw | tee /tmp/gen.out.models.4

# detokenize + eval bleu
cat /tmp/gen.out.models.4 | grep ^H | sort -nr -k1.2 | cut -f3- | ./mosesdecoder/scripts/tokenizer/detokenizer.perl | sacrebleu -t wmt19 -l $PAIR
