#!/usr/bin/env python
# coding: utf-8

# this script was used to compare the state_dict after conversion
#
# it's not quite runnable at the moment as it was comparing against a temp dump
# of a state_dict that was being ported, but you get the idea

import sys
sys.path.insert(0, "/code/huggingface/transformers-fair-wmt/src")

from transformers.tokenization_fsmt import FSMTTokenizer
from transformers.modeling_fsmt import FSMTForConditionalGeneration
mname = 'data/wmt19-ru-en'

import torch
def compare_state_dicts(d1, d2, cmp_func=torch.equal):
    ok = 1
    for k in sorted( set(d1.keys()) | set(d2.keys()) ):
        if k in d1 and k in d2:
            if not cmp_func(d1[k], d2[k]):
                ok = 0
                print(f"! Key {k} mismatches:")
                if d1[k].shape != d2[k].shape:
                    print(f"- Shapes: \n{d1[k].shape}\n{d2[k].shape}")
                print(f"- Values:\n{d1[k]}\n{d2[k]}\n")
        else:
            ok = 0
            which = "1st" if k in d2 else "2nd"
            print(f"{which} dict doesn't have key {k}\n")
    if ok:
        print('Models match')

tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

# this fixes the problem
import torch
d2 = torch.load("/tmp/new.pt")
compare_state_dicts(model.state_dict(), d2)
#model.load_state_dict(d2)
#model.load_state_dict(torch.load("/tmp/new.pt"))

print("Wrong shape?", model.state_dict()['model.decoder.embed_tokens.weight'].shape)
sentence = "Машинное обучение - это здорово! Ты молодец."

input_ids = tokenizer.encode(sentence, return_tensors='pt')
print(input_ids)
outputs = model.generate(input_ids)#, num_beams=5)
print("Outputs")
print(outputs)
for output in outputs:
    decoded = tokenizer.decode(output, skip_special_tokens=True)
    print(decoded)



# compare_state_dicts(model.state_dict(), model.state_dict())
