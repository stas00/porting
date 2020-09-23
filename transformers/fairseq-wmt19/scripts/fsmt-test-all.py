#!/usr/bin/env python
# coding: utf-8

# this script validates that the 3 languages all translate into each other as expected
# this was a quick prototype for an integration test

import sys
sys.path.insert(0, "/code/huggingface/transformers-fair-wmt/src")

import logging
logging.disable(logging.INFO) # disable INFO and DEBUG logger everywhere

from transformers import FSMTTokenizer, FSMTForConditionalGeneration
from transformers.modeling_fsmt import FSMTForConditionalGeneration

text = {
    "en": "Machine learning is great, isn't it?",
    "ru": "Машинное обучение - это здорово, не так ли?",
    "de": "Maschinelles Lernen ist großartig, oder?",
}

pairs = [
    ["en", "ru"],
    ["ru", "en"],
    ["en", "de"],
    ["de", "en"],
]

for src, tgt in pairs:
    print(f"Testing {src} -> {tgt}")

    # to switch to local model
    #mname = "/code/huggingface/transformers-fair-wmt/data/wmt19-{src}-{tgt}"
    # s3 uploaded model
    mname = f"stas/wmt19-{src}-{tgt}"

    src_sentence = text[src]
    tgt_sentence = text[tgt]

    tokenizer = FSMTTokenizer.from_pretrained(mname)
    model = FSMTForConditionalGeneration.from_pretrained(mname)

    encoded = tokenizer.encode(src_sentence, return_tensors='pt')
    #print(encoded)
    
    outputs = model.generate(encoded)
    #print(outputs)
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #print(decoded)
    assert decoded == tgt_sentence, f"\n\ngot: {decoded}\nexp: {tgt_sentence}\n"
