#!/usr/bin/env python

# this script demonstrates paraphrasing in fairseq wmt19 transformer

import sys
sys.path.insert(0, "/code/github/00nlp/fairseq")

import logging
logging.disable(logging.WARNING) # disable INFO and DEBUG logger everywhere

import torch

def translate(src_lang, tgt_lang, text):
    mname = f"transformer.wmt19.{src_lang}-{tgt_lang}"
    checkpoint_file = 'model4.pt'
    #checkpoint_file = 'model1.pt:model2.pt:model3.pt:model4.pt'
    model = torch.hub.load('pytorch/fairseq', mname, checkpoint_file=checkpoint_file, tokenizer='moses', bpe='fastbpe')

    encoded = model.encode(text)
    # print(encoded)

    output = model.generate(encoded, beam=5)[0]["tokens"]
    # print(output)

    decoded = model.decode(output)
    #print(decoded)
    return decoded

def paraphrase(src_lang, tgt_lang, text):
    return translate(tgt_lang, src_lang, translate(src_lang, tgt_lang, text))

#text = """Here's a little song I wrote. You might want to sing it note for note. Don't worry, be happy."""

text = "Every morning when I wake up, I experience an exquisite joy - the joy of being Salvador Dalí - and I ask myself in rapture: What wonderful things is this Salvador Dalí going to accomplish today?"

en_ru = paraphrase('en', 'ru', text)
en_de = paraphrase('en', 'de', text)
# print together to avoid the logger noise :(
print("Paraphrasing:")
print(f"en      : {text}")
print(f"en-ru-en: {en_ru}")
print(f"en-de-en: {en_de}")

# Paraphrasing:
# en      : Here's a little song I wrote. You might want to sing it note for note. Don't worry, be happy.
# en-ru-en: Here is a little song I wrote. You may want to put it on the record. Don't worry, be happy.
# en-de-en: Here is a little song I wrote. Maybe you want to sing it note by note. Don't worry, be happy.