#!/usr/bin/env python
# coding: utf-8

# this script prepares data for pegasus/bigpatent eval

# see process.txt for instructions
# pip install pegasus
# 
# ln -s ~/nltk_data /home/stas/anaconda3/envs/main-38/lib/python3.8/site-packages/nltk/

# 1. ./process.py

from pegasus.data import all_datasets
from pathlib import Path

input_pattern = "tfds:big_patent/all"
split = "test"
ds_test = all_datasets.get_dataset(input_pattern + "-" + split, shuffle_files=False)

save_path = Path("data")
save_path.mkdir(parents=True, exist_ok=True)
src_path = save_path / "test.source"
tgt_path = save_path / "test.target"
with open(src_path, 'wt') as src_file, open(tgt_path, 'wt') as tgt_file:
    for i, d in enumerate(ds_test):
        src = d["inputs"].numpy().decode()
        tgt = d["targets"].numpy().decode()
        src_len, tgt_len = len(src), len(tgt)
        
        #  a threshold is used to remove short articles with long summaries as well as articles with no summary
        if src_len and tgt_len and tgt_len < 0.75*src_len:
            src = src.replace('\n', '<n>')
            tgt = tgt.replace('\n', '<n>')        
            src_file.write(src + '\n')
            tgt_file.write(tgt + '\n')        

print(f"Generated {src_path}")
print(f"Generated {tgt_path}")
