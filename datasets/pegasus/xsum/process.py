#!/usr/bin/env python
# coding: utf-8

# this script prepares data for pegasus/xsum eval
# 0. install pegasus (see README.md)
# 0. pip install datasets
# 1. ./process.py

# this one is not working in tfds:xsum, so using datasets' xsum

from datasets import load_dataset
from pathlib import Path

ds_all = load_dataset("xsum")

TEST_ONLY = True

splits = ['test'] if TEST_ONLY else ['test', 'validation', 'train']

for split in splits:

    ds = ds_all[split]

    save_path = Path("data")
    save_path.mkdir(parents=True, exist_ok=True)
    src_path = save_path / f"{split}.source"
    tgt_path = save_path / f"{split}.target"
    with open(src_path, 'wt') as src_file, open(tgt_path, 'wt') as tgt_file:
        for i, d in enumerate(ds):
            src, tgt = d['document'], d['summary']
            src_len, tgt_len = len(src), len(tgt)

            #  remove articles with no summary
            if src_len and tgt_len:
                src = src.replace('\n', '<n>')
                tgt = tgt.replace('\n', '<n>')        
                src_file.write(src + '\n')
                tgt_file.write(tgt + '\n')        

    print(f"Generated {src_path}")
    print(f"Generated {tgt_path}")
