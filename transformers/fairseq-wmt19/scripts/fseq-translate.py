#!/usr/bin/env python

# this script demonstrates 5 beam search translation in fairseq wmt19 transformer

import sys
sys.path.insert(0, "/code/github/00nlp/fairseq")

import torch

text = [
    """Welsh AMs worried about 'looking like muppets'""",
    """There is consternation among some AMs at a suggestion their title should change to MWPs (Member of the Welsh Parliament).""",
    """It has arisen because of plans to change the name of the assembly to the Welsh Parliament.""",
    """AMs across the political spectrum are worried it could invite ridicule.""",
    """One Labour AM said his group was concerned "it rhymes with Twp and Pwp.""""",
    ]

text = [text[0]]

text = ["Machine learning is great, isn't it?"]

pairs = [["en", "ru"], ]
pairs = [["en", "de"], ]
num_beams = 5

for src, tgt in pairs:
    print(f"Testing {src} -> {tgt}")

    mname = f"transformer.wmt16.{src}-{tgt}"
    checkpoint_file = 'model.pt'
    model = torch.hub.load('pytorch/fairseq', mname, checkpoint_file=checkpoint_file)


    #mname = f"transformer.wmt19.{src}-{tgt}"
    #checkpoint_file = 'model1.pt:model2.pt:model3.pt:model4.pt'
    #model = torch.hub.load('pytorch/fairseq', mname, checkpoint_file=checkpoint_file, tokenizer='moses', bpe='fastbpe')

    # mname = f"/code/huggingface/transformers-fair-wmt/data/wmt16-en-de-dist-12-1"
    # checkpoint_file = 'checkpoint_best.pt'
    # model = torch.hub.load('pytorch/fairseq', mname, checkpoint_file=checkpoint_file)



    for s in text:
        encoded = model.encode(s)
        #print(encoded)

        # no beam search
        # print("*** 1 beam")
        # output = model.generate(encoded, beam=1)[0]['tokens']
        # print(output.tolist())
        # decoded = model.decode(output)
        # print(decoded)
        
        # beam search
        print(f"*** {num_beams} beams")
        outputs = model.generate(encoded, beam=num_beams)
        for i, output in enumerate(outputs):
            i += 1
            output = output['tokens']
            print(f"{i}: {output.tolist()}")
            decoded = model.decode(output)
            print(f"{i}: {decoded}")



# a way to run a custom model that is not on torch.hub
# from fairseq.models.transformer import TransformerModel
# model_dir = <PATH_TO_MY_MODEL_DIR>
# en2de = TransformerModel.from_pretrained(model_dir, checkpoint_file='checkpoint_top5_average.pt',
#     data_name_or_path=model_dir, bpe='subword_nmt', bpe_codes=model_dir+'bpecodes', tokenizer='moses')
# text = 'Machine learning is great, isn't it?'
# en2de.translate(text)
# text = ["Machine learning is great, isn't it?", "Everyone should learn machine learning."]
# en2de.translate(text)