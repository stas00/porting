#!/usr/bin/env python
# coding: utf-8

# this script just does a decode of outputs codes

import sys
sys.path.insert(0, "/code/huggingface/transformers-fair-wmt/src")

from transformers.tokenization_fsmt import FSMTTokenizer
tokenizer = FSMTTokenizer.from_pretrained('stas/wmt19-ru-en')

outputs = [[    2,  5494,  3221,    21,  1054,   427,   739,  4952,    11,   700,  18128,     7,     2]]
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)

