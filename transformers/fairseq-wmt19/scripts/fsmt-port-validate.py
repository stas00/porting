#!/usr/bin/env python
# coding: utf-8

# this script runs a side-by-side translation via fairseq and via transformers
# and compares that each stage - encoding, generation, and decoding outputs -
# all match
#
# there are some small variations in how the two systems generate outputs, so
# the script takes care to adjust outputs so that the two are comparable
#
# the eval data is taken from http://matrix.statmt.org/matrix (select
# newstest2019 in the dropdown), via `sacrebleu` (https://github.com/mjpost/sacrebleu)

import sys
sys.path.insert(0, "/code/huggingface/transformers-fair-wmt/src")

import logging
logging.disable(logging.INFO) # disable INFO and DEBUG logger everywhere

from transformers.tokenization_fsmt import FSMTTokenizer
from transformers.modeling_fsmt import FSMTForConditionalGeneration
import torch
import subprocess

text = {
    'en':     [
    """Welsh AMs worried about 'looking like muppets'""",
    """There is consternation among some AMs at a suggestion their title should change to MWPs (Member of the Welsh Parliament).""",
    """It has arisen because of plans to change the name of the assembly to the Welsh Parliament.""",
    """AMs across the political spectrum are worried it could invite ridicule.""",
    """One Labour AM said his group was concerned "it rhymes with Twp and Pwp.\"""",
    """For readers outside of Wales: In Welsh twp means daft and pwp means poo.""",
    """A Plaid AM said the group as a whole was "not happy" and has suggested alternatives.""",
    """A Welsh Conservative said his group was "open minded" about the name change, but noted it was a short verbal hop from MWP to Muppet.""",
    """In this context The Welsh letter w is pronounced similarly to the Yorkshire English pronunciation of the letter u.""",
    """The Assembly Commission, which is currently drafting legislation to introduce the name changes, said: "The final decision on any descriptors of what Assembly Members are called will of course be a matter for the members themselves.\"""",
    """The Government of Wales Act 2017 gave the Welsh assembly the power to change its name.""",
    """In June, the Commission published the results of a public consultation on the proposals which found broad support for calling the assembly a Welsh Parliament.""",
    """On the matter of the AMs' title, the Commission favoured Welsh Parliament Members or WMPs, but the MWP option received the most support in a public consultation.""",
    """AMs are apparently suggesting alternative options, but the struggle to reach consensus could be a headache for the Presiding Officer, Elin Jones, who is expected to submit draft legislation on the changes within weeks.""",
    """The legislation on the reforms will include other changes to the way the assembly works, including rules on disqualification of AMs and the design of the committee system.""",
    ],
    'ru': [
    """Названо число готовящихся к отправке в Донбасс новобранцев из Украины""",
    """Официальный представитель Народной милиции самопровозглашенной Луганской Народной Республики (ЛНР) Андрей Марочко заявил, что зимой 2018-2019 года Украина направит в Донбасс не менее 3 тыс. новобранцев.""",
    """По его словам, таким образом Киев планирует "хоть как-то доукомплектовать подразделения".""",
    """"Нежелание граждан Украины проходить службу в рядах ВС Украины, массовые увольнения привели к низкой укомплектованности подразделений", - рассказал Марочко, которого цитирует "РИА Новости".""",
    """Он также не исключил, что реальные цифры призванных в армию украинцев могут быть увеличены в случае необходимости.""",
    """В 2014-2017 годах Киев начал так называемую антитеррористическую операцию (АТО), которую позже сменили на операцию объединенных сил (ООС).""",
    """Предполагалось, что эта мера приведет к усилению роли украинских силовиков в урегулировании ситуации.""",
    """В конце августа 2018 года ситуация в Донбассе обострилась из-за убийства главы ДНР Александра Захарченко.""",
    """Власти ДНР квалифицировали произошедшее как теракт.""",
    """В ходе расследования уголовного дела были задержаны несколько человек, причастных ко взрыву.""",
    """По предварительным данным, они подтвердили, что диверсию организовали украинские спецслужбы.""",
    """Власти США заставили Маска покинуть пост председателя правления Tesla""",
    """Американский бизнесмен Илон Маск покинет должность председателя совета директоров основанной им компании Tesla по требованию властей США.""",
    """Для урегулирования претензий федеральной Комиссии по ценным бумагам и биржам предприниматель также выплатит $20 млн.""",
    """По данным телеканала CNBC, досудебная договоренность с ведомством позволяет Маску остаться на руководящем посту в Tesla.""",
    """There is consternation among some AMs at a suggestion their title should change to MWPs (Member of the Welsh Parliament).""",
    ],
    'de': [
    """Schöne Münchnerin 2018: Schöne Münchnerin 2018 in Hvar: Neun Dates""",
    """Von az, aktualisiert am 04.05.2018 um 11:11""",
    """Ja, sie will...""",
    """"Schöne Münchnerin" 2018 werden!""",
    """Am Nachmittag wartet erneut eine Überraschung auf unsere Kandidatinnen: sie werden das romantische Candlelight-Shooting vor der MY SOLARIS nicht alleine bestreiten, sondern an der Seite von Male-Model Fabian!""",
    """Hvar - Flirten, kokettieren, verführen - keine einfachen Aufgaben für unsere Mädchen.""",
    """Insbesondere dann, wenn in Deutschland ein Freund wartet.""",
    """Dennoch liefern die neun "Schöne Münchnerin"-Kandidatinnen beim Shooting mit People-Fotograf Tuan ab und trotzen Wind, Gischt und Regen wie echte Profis.""",
    """Das romantische Shooting gibt's im Video:""",
    """DFB-Chef Grindel sieht keine Grundlage für Özil-Comeback""",
    """Berlin DFB-Präsident Reinhard Grindel sieht derzeit keine Grundlage für eine Rückkehr von Mesut Özil in die Nationalmannschaft.""",
    """"Das ist eine Frage, die natürlich davon abhängt, dass man einmal ins Gespräch kommt, dass man mit ihm auch darüber spricht, warum er das eine oder andere offenbar so empfunden hat, wie das in seinem Statement niedergelegt ist", sagte Grindel im Fußball-Podcast "Phrasenmäher" der "Bild-Zeitung.""",
    """Trotz der nun monatelangen Debatte um die Fotos von Özil mit dem türkischen Präsidenten Recep Tayyip Erdogan bedauere er den Rücktritt des 92-fachen Nationalspielers Özil.""",
    """"Ich hätte mich gefreut, wenn Mesut Özil weiter für Deutschland gespielt hätte.""",
    """Ich bin auch bis zu einem gewissen Zeitpunkt davon ausgegangen, dass das so sein würde", sagte Grindel.""",
    ],

}
pairs = [["ru", "en"],["en", "de"],["de", "en"],["en", "ru"], ]

#text = [text[0]]

#pairs = [["en", "ru"]]


# text = {
#         'en':     [
#             """One Labour AM said his group was concerned "it rhymes with Twp and Pwp.\"""",
#             """AMs across the political spectrum are worried it could invite ridicule.""",
#             #            """This is good. This is bad. This is good. This is bad. What do you think?""",
# #            """Here's a little song I wrote. You might want to sing it note for note. Don't worry, be happy. In every life we have some trouble. But when you worry you make it double. Don't worry, be happy. Don't worry, be happy now."""
#     ],
# }

# enable to compare each entry in the beam and not just the first one
check_all_beams = False
# enable to do a massive check of 2000 entries per languge, instead of just 10 that are hardcoded in this script
do_massive_check = False


beams = 5
print(f"Using {beams} beams")

def get_all_data(pairs):
    text = {}
    for src, tgt in pairs:
        pair = f"{src}-{tgt}"
        cmd = f"sacrebleu -t wmt19 -l {pair} --echo src".split()
        out = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8').splitlines()
        text[src] = out
        # print(out)
    return text


if do_massive_check:
    text = get_all_data(pairs)

def check_beam(j, fs_outputs, tf_outputs):
    fs_output = fs_outputs[j]['tokens'].tolist()
    tf_output = tf_outputs[j].tolist()
    #print(fs_output, tf_output, sep="\n")
    if 2 in fs_output:
        fs_output = fs_output[:fs_output.index(2)] # remove trailing 1 [..., 2, 1]
    tf_output = tf_output[1:] # remove leading 2 [2, ...]
    if 2 in tf_output:
        tf_output = tf_output[:tf_output.index(2)] # remove trailing 1 [..., 2, 1]
    if fs_output != tf_output:
        ok = False
        print(f"{j}: Generation mismatch for: {s}")
        print(fs_output, tf_output, sep="\n")
    #print(fs_output)
    #print(tf_output)
    fs_decoded = fs_model.decode(fs_output)
    fs_decoded = fs_decoded.replace(" 's", "'s") # bug in fairseq when the original word with 's doesn't get translated
    tf_decoded = tf_tokenizer.decode(tf_output, skip_special_tokens=True)
    if fs_decoded != tf_decoded:
        ok = False
        print(f"{j}: Decoding mismatch for: {s}")
        print(fs_decoded, tf_decoded, sep="\n")
    # print(f"{i}: {fs_decoded}")
    # print(f"{i}: {tf_decoded}")


for src, tgt in pairs:
    print(f"Testing {src} -> {tgt}")

    t = text[src]
    # to switch to local model
    # mname = f"data/wmt19-{src}-{tgt}"
    # mname_tok = f"wmt19-{src}-{tgt}"

    fs_mname = f"transformer.wmt19.{src}-{tgt}"
    checkpoint_file = 'model4.pt'
    #checkpoint_file = 'model1.pt:model2.pt:model3.pt:model4.pt'
    fs_model = torch.hub.load('pytorch/fairseq', fs_mname, checkpoint_file=checkpoint_file, tokenizer='moses', bpe='fastbpe')

    # s3 uploaded model
    tf_mname = f"stas/wmt19-{src}-{tgt}"
    tf_tokenizer = FSMTTokenizer.from_pretrained(tf_mname)
    tf_model = FSMTForConditionalGeneration.from_pretrained(tf_mname)

    for i, s in enumerate(t):
        #print(s)
        ok = True
        fs_input_ids = fs_model.encode(s)
        tf_input_ids = tf_tokenizer.encode(s, return_tensors='pt')
        #print(tf_input_ids[0].tolist())
        if fs_input_ids.tolist() != tf_input_ids[0].tolist():
            ok = False
            print(f"Encoding mismatch for: {s}")
            print(fs_input_ids.tolist(), tf_input_ids[0].tolist(), sep="\n")
        #print("*** 5 beams")
        fs_outputs = fs_model.generate(fs_input_ids, beam=beams, verbose=True)
        tf_outputs = tf_model.generate(tf_input_ids, num_beams=beams, num_return_sequences=beams, early_stopping=True)
        if check_all_beams:
            for j in range(beams):
                check_beam(j, fs_outputs, tf_outputs)
        else:
            check_beam(0, fs_outputs, tf_outputs)

        print(f"{'✓' if ok else '✗'} input {i+1:02d}")

