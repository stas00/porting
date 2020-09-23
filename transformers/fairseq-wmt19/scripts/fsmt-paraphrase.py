#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, "/code/huggingface/transformers-fair-wmt/src")

import logging
logging.disable(logging.INFO) # disable INFO and DEBUG logger everywhere

from transformers.tokenization_fsmt import FSMTTokenizer
from transformers.modeling_fsmt import FSMTForConditionalGeneration

def translate(src, tgt, text):
    # to switch to local model
    #mname = "/code/huggingface/transformers-fair-wmt/data/wmt19-{src}-{tgt}"
    # s3 uploaded model
    mname = f"stas/wmt19-{src}-{tgt}"
    tokenizer = FSMTTokenizer.from_pretrained(mname)
    model = FSMTForConditionalGeneration.from_pretrained(mname)

    encoded = tokenizer.encode(text, return_tensors='pt')
    # print(encoded)

    output = model.generate(encoded, num_beams=5, early_stopping=True)[0]
    # print(output)

    decoded = tokenizer.decode(output, skip_special_tokens=True)
    #print(decoded)
    return decoded

def paraphrase(src, tgt, text):
    return translate(tgt, src, translate(src, tgt, text))

#text = """Here's a little song I wrote. You might want to sing it note for note. Don't worry, be happy. In every life we have some trouble. But when you worry you make it double. Don't worry, be happy. Don't worry, be happy now."""

text = "Every morning when I wake up, I experience an exquisite joy - the joy of being Salvador Dalí - and I ask myself in rapture: What wonderful things is this Salvador Dalí going to accomplish today?"

en_ru = paraphrase('en', 'ru', text)
en_de = paraphrase('en', 'de', text)
# print together to avoid the logger noise :(
print("Paraphrasing:")
print(f"en      : {text}")
print(f"en-ru-en: {en_ru}")
print(f"en-de-en: {en_de}")

# Paraphrasing:
# en      : Every morning when I wake up, I experience an exquisite joy - the joy of being Salvador Dalí - and I ask myself in rapture: What wonderful things is this Salvador Dalí going to accomplish today?
# en-ru-en: Every morning when I wake up, I have a delightful joy - the joy of being Salvador Dali - and I ask myself in delight: What wonderful things is this Salvador Dali going to do today?
# en-de-en: Every morning when I wake up, I experience an extraordinary joy - the joy of being Salvador Dalí - and I wonder with delight: what wonderful things will this Salvador Dalí do today?
