#!/usr/bin/env python
# coding: utf-8

# this script prepares data for pegasus/billsum eval

# 0. pip install pegasus
# 1. ./process-all.py

from pegasus.data import all_datasets
from pathlib import Path

dss = dict(
    aeslc="tfds:aeslc",
#    bigpatent_all="tfds:big_patent/all",
#    bigpatent_y="tfds:big_patent/y",
    billsum="tfds_transformed:billsum",
    cnn_dailymail="tfds:cnn_dailymail/plain_text",
#    gigaword="tfds:gigaword",
    multi_news="tfds:multi_news",
#    newsroom="tfds:newsroom",
#    newsroom_abstractive="tfds_transformed:newsroom_abstractive",
    reddit_tifu_short="tfds_transformed:reddit_tifu/short",
    reddit_tifu_long="tfds_transformed:reddit_tifu/long",
    arxiv="tfds:scientific_papers/arxiv",
    pubmed="tfds:scientific_papers/pubmed",
    wikihow="tfds:wikihow/all",
#    xsum="tfds:xsum",
)

RULE75 = False

for dataset_name, input_pattern in dss.items():
    for split in ['test', 'validation', 'train']:
        ds = all_datasets.get_dataset(input_pattern + "-" + split, shuffle_files=False)

        save_path = Path(f"data/{dataset_name}")
        save_path.mkdir(parents=True, exist_ok=True)
        src_path = save_path / f"{split}.source"
        tgt_path = save_path / f"{split}.target"
        with open(src_path, 'wt') as src_file, open(tgt_path, 'wt') as tgt_file:
            for i, d in enumerate(ds):
                src = d["inputs"].numpy().decode()
                tgt = d["targets"].numpy().decode()
                src_len, tgt_len = len(src), len(tgt)

                # sanity check: skip broken entries with len 0
                if not src_len or not tgt_len:
                    continue

                # a threshold to remove articles with summaries almost as long as the source
                if RULE75 and tgt_len >= 0.75*src_len:
                    continue
                
                src = src.replace('\n', '<n>')
                tgt = tgt.replace('\n', '<n>')        
                src_file.write(src + '\n')
                tgt_file.write(tgt + '\n')        

        print(f"Generated {src_path}")
        print(f"Generated {tgt_path}")
