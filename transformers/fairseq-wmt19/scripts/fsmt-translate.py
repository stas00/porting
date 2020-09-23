#!/usr/bin/env python

# this script demonstrates 5 beam search translation in fsmt

import sys
sys.path.insert(0, "/code/huggingface/transformers-fair-wmt/src")

from transformers import FSMTTokenizer, FSMTForConditionalGeneration

pair = 'ru-en'
#pair = 'en-ru'
#pair = 'de-en'

# to switch to local model
mname = f"/code/huggingface/transformers-fair-wmt/data/wmt19-{pair}"

#pair = 'en-de'
#mname = "/code/huggingface/transformers-fair-wmt/data/wmt16-en-de-12-1"

# s3 uploaded model
#mname = f"stas/wmt19-{pair}"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname)

# text = """
# Каждый день я просыпаюсь полный удивления.
# Еще один шанс в многообещающий день.
# И я всегда говорю какая чудесная жизнь.
# """
#text = """я думаю, следовательно, я существую"""
#text = """я люблю, следовательно, я существую"""

text = "Machine learning is great, isn't it?"

#text = """One Labour AM said his group was concerned "it rhymes with Twp and Pwp.\""""
#text = """One AM said his group was concerned "it rhymes with Twp and Pwp.\""""

#print(text)

encoded = tokenizer.encode(text, return_tensors='pt')
#print(encoded[0].tolist())

outputs = model.generate(encoded, num_beams=5, num_return_sequences=5, early_stopping=True)
for i, output in enumerate(outputs):
    i += 1
    print(f"{i}: {output.tolist()}")
    
    decoded = tokenizer.decode(output, skip_special_tokens=True)
    print(f"{i}: {decoded}")
