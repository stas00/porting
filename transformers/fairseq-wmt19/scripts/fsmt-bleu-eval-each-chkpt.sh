#!/bin/bash

# this script converts one of 4 fairseq checkpoints into the transformers format
# and then evaluates it - we want to see which of the 4 checkpoints gives the
# highest BLEU score

# once elsewhere his has been run to get all the data in place

# export ROOT=/code/huggingface/transformers-fair-wmt
# cd $ROOT
# mkdir data

# # get data (run once)
# wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz
# wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz
# wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz
# wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz
# tar -xvzf wmt19.en-de.joined-dict.ensemble.tar.gz
# tar -xvzf wmt19.de-en.joined-dict.ensemble.tar.gz
# tar -xvzf wmt19.en-ru.ensemble.tar.gz
# tar -xvzf wmt19.ru-en.ensemble.tar.gz


# we are working from the root of the transformers clone

export BS=8
# set to 5 for a quick test run, set to 2000 to eval all available records
export OBJS=2000
# at the end we want NUM_BEAMS=50 (as that's what fairseq used in their eval)
export NUM_BEAMS=50

pairs=(ru-en en-ru en-de de-en)
for pair in "${pairs[@]}"
do
    export PAIR=$pair
    export DATA_DIR=data/$PAIR
    export SAVE_DIR=data/$PAIR
    mkdir -p $DATA_DIR
    sacrebleu -t wmt19 -l $PAIR --echo src | head -$OBJS > $DATA_DIR/val.source
    sacrebleu -t wmt19 -l $PAIR --echo ref | head -$OBJS > $DATA_DIR/val.target

    if [[ $pair =~ "ru" ]]
    then
        subdir=ensemble # ru folders
    else
        subdir=joined-dict.ensemble # de data folders are different
    fi

    END=4;
    for i in $(seq 1 $END);
    do
        model=model$i.pt;
        CHKPT=$model PYTHONPATH="src" python src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py --fsmt_checkpoint_path data/wmt19.$PAIR.$subdir --pytorch_dump_folder_path data/wmt19-$PAIR > log.$PAIR-$model 2>&1
        echo "###" $PAIR $model num_beams=$NUM_BEAMS objs=$OBJS
        PYTHONPATH="src:examples/seq2seq" python examples/seq2seq/run_eval.py /code/huggingface/transformers-fair-wmt/data/wmt19-$PAIR $DATA_DIR/val.source $SAVE_DIR/test_translations.txt --reference_path $DATA_DIR/val.target --score_path $SAVE_DIR/test_bleu.json --bs $BS --task translation 2> /dev/null
    done

    echo
    echo
done
