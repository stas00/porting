#!/usr/bin/env python
# coding: utf-8

# this script prepares data for pegasus/wikihow eval

# 0. pip install datasets
# 1. manually download https://ucsb.app.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358 and save to some path
# 2. adjust data_dir below to where it was downloaded
# 3. run ./process.py

from datasets import load_dataset
from pathlib import Path

data_dir = "/hf/pegasus-datasets/pubmed/"

ds_all = load_dataset("pubmed", 'all', data_dir=data_dir)

ds_test = ds_all['test']

save_path = Path("data")
save_path.mkdir(parents=True, exist_ok=True)
src_path = save_path / "test.source"
tgt_path = save_path / "test.target"
with open(src_path, 'wt') as src_file, open(tgt_path, 'wt') as tgt_file:
    for i, e in enumerate(ds_test):
        src, tgt = e['text'], e['headline']
        src_len, tgt_len = len(src), len(tgt)
        
        #  a threshold is used to remove short articles with long summaries as well as articles with no summary
        if src_len and tgt_len and tgt_len < 0.75*src_len:
            src = src.replace('\n', '<n>')
            tgt = tgt.replace('\n', '<n>')        
            src_file.write(src + '\n')
            tgt_file.write(tgt + '\n')        

print(f"Generated {src_path}")
print(f"Generated {tgt_path}")
