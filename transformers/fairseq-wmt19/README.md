# Porting fairseq wmt19 translation system to transformers

This is an attempt to documented how [fairseq wmt19 translation system](https://github.com/pytorch/fairseq/tree/master/examples/wmt19) was ported to [`transformers`](https://github.com/huggingface/transformers/).

I was looking for some interesting project to work on and [Sam Shleifer](https://github.com/sshleifer) suggested I work on [porting a high quality translator](https://github.com/huggingface/transformers/issues/5419).

I read the short paper: [Facebook FAIR's WMT19 News Translation Task Submission](https://arxiv.org/abs/1907.06616) that describes the original system and decided to give it a try.

I had no idea how to approach this complex problem and Sam helped me to [break it down to smaller tasks](https://github.com/huggingface/transformers/issues/5419), which was of a great help.

I chose to work with the en-ru/ru-en models during porting as I speak both languages. It'd have been much more difficult to work with de-en/en-de as I don't speak German, and being able to evaluate the translation quality by just reading and making sense of the outputs at the advanced stages of the porting process saved me a ton of time.

Also, as I did the porting with the en-ru/ru-en models, I was totally unaware that the de-en/en-de models used a merged vocabulary, whereas the former used 2 separate vocabularies of different sizes. So once I did the more complicated work of supported 2 separate vocabularies, it was trivial to get the merged vocabulary to work.

## Let's cheat

The first step was to cheat, of course. Why make a complex effort when one can make a little one. So I wrote a [short notebook](./nbs/cheat.ipynb) that in a few lines of code provided a proxy to fairseq and emulated `transformers` API. 

If no other things but basic translation was required, this would have been enough. But, of course, we wanted to have the full porting, so after having this small victory, I moved onto much harder things.

## Installations

For the sake of this article let's assume that we work under `~/porting`, so let's create this directory:
```
mkdir ~/porting
cd ~/porting
```

We need to install a few things for this work:

```
# install fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install -e .
# install mosesdecoder under fairseq
git clone https://github.com/moses-smt/mosesdecoder
# install fastBPE under fairseq
git clone git@github.com:glample/fastBPE.git
cd fastBPE; g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast; cd -
cd -

# install transformers
git clone https://github.com/huggingface/transformers/
pip install -e .[dev]

```

## Files

To get an idea of what needs to be done code-wise, the following files need to be created when the work is completed:

* [src/transformers/configuration_fsmt.py](https://github.com/huggingface/transformers/blob/129fdae04033fe4adfe013b734deaec6ec34ae2e/src/transformers/configuration_fsmt.py) -  a short configuration class.
* [src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py](https://github.com/huggingface/transformers/blob/129fdae04033fe4adfe013b734deaec6ec34ae2e/src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py) - a complex conversion script. 
* [src/transformers/modeling_fsmt.py](https://github.com/huggingface/transformers/blob/129fdae04033fe4adfe013b734deaec6ec34ae2e/src/transformers/modeling_fsmt.py) - this is where the model architecture is implemented.
* [src/transformers/tokenization_fsmt.py](https://github.com/huggingface/transformers/blob/129fdae04033fe4adfe013b734deaec6ec34ae2e/src/transformers/tokenization_fsmt.py) - a tokenizer code
* [tests/test_modeling_fsmt.py](https://github.com/huggingface/transformers/blob/129fdae04033fe4adfe013b734deaec6ec34ae2e/src/transformers/tests/test_modeling_fsmt.py) - model tests
* [tests/test_tokenization_fsmt.py](https://github.com/huggingface/transformers/blob/129fdae04033fe4adfe013b734deaec6ec34ae2e/src/transformers/tests/test_tokenization_fsmt.py) - tokenizer tests
* [docs/source/model_doc/fsmt.rst](https://github.com/huggingface/transformers/blob/129fdae04033fe4adfe013b734deaec6ec34ae2e/src/transformers/docs/source/model_doc/fsmt.rst) - a doc file

there are other files that need to be modified as well, we will talk about those towards the end.


## Conversion

One of the most important parts of the porting process is creating the conversion script. It will take all the available source data provided by the original developer of the model, which includes checkpoint with pre-trained weights, model and training configuration details, dictionaries and tokenizer support files, and convert them into a new set of files supported by `transformers`. You will find the final script here: [src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py](https://github.com/huggingface/transformers/blob/129fdae04033fe4adfe013b734deaec6ec34ae2e/src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py)

I started by copying one of the existing conversion scripts, gutted most of it out and then gradually added parts to it as I was porting part by part.

During the development I was testing all my code against a local copy of the converted model, and only at the very end when everything was ready I uploaded it to s3 and then continued testing against this version.

## fairseq model and its support files

Let's first look at what data we get with the fairseq model. We are going to use the convenient `torch.hub` API, which makes it very easy to apply the models.
```
import torch
torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru', checkpoint_file='model4.pt', tokenizer='moses', bpe='fastbpe')
```
This code downloads the model and its support files. To look inside we have to hunt down the downloaded files in the `~/.cache` folder.

```
ls -1 ~/.cache/torch/hub/pytorch_fairseq/
```
shows:
```
15bca559d0277eb5c17149cc7e808459c6e307e5dfbb296d0cf1cfe89bb665d7.ded47c1b3054e7b2d78c0b86297f36a170b7d2e7980d8c29003634eb58d973d9
15bca559d0277eb5c17149cc7e808459c6e307e5dfbb296d0cf1cfe89bb665d7.ded47c1b3054e7b2d78c0b86297f36a170b7d2e7980d8c29003634eb58d973d9.json
```

Let's make a symlink so that we can easily refer to that obscured cache folder down the road:

```
ln -s /code/data/cache/torch/hub/pytorch_fairseq/15bca559d0277eb5c17149cc7e808459c6e307e5dfbb296d0cf1cfe89bb665d7.ded47c1b3054e7b2d78c0b86297f36a170b7d2e7980d8c29003634eb58d973d9 ~/porting/pytorch_fairseq_model
```

Note: the path could be different when you try it yourself, since the hash value of the model could change. You will find the right one in `~/.cache/torch/hub/pytorch_fairseq/`

If we look inside that folder:
```
ls -l ~/porting/pytorch_fairseq_model/
total 13646584
-rw-rw-r-- 1 stas stas     532048 Sep  8 21:29 bpecodes
-rw-rw-r-- 1 stas stas     351706 Sep  8 21:29 dict.en.txt
-rw-rw-r-- 1 stas stas     515506 Sep  8 21:29 dict.ru.txt
-rw-rw-r-- 1 stas stas 3493170533 Sep  8 21:28 model1.pt
-rw-rw-r-- 1 stas stas 3493170532 Sep  8 21:28 model2.pt
-rw-rw-r-- 1 stas stas 3493170374 Sep  8 21:28 model3.pt
-rw-rw-r-- 1 stas stas 3493170386 Sep  8 21:29 model4.pt
```
we have:
1. `model*.pt` - 4 checkpoints (pytorch `state_dict` with all the pretrained weights, and various other things)
2. `dict.*.txt` - source and target dictionaries
3. `bpecodes` - special map file for BPE work

We are going to investigate each of these files in the following sections.

## How translation systems work

Here is a bit of an introduction to how a computer translates text nowadays.

Computers can't read text, but can only handle numbers. So when working with text we have to map one or more letters into numbers, and hand those to a computer program. When the program completes it too returns  numbers, which we need to convert back into text. 

Let's start with two sentences in Russian and English:
```
я  люблю следовательно я  существую
10 11    12            10 13

I  love therefore I  am
20 21   22        20 23
```

Let's assign a unique number to each word. The numbers starting with 10 are a map of Russian words to unique numbers. The numbers starting with 20 are a different map for English words. If you don't speak Russian, you can still see that the word `я` repeats twice in the sentence and it gets the same number 10 associated with it. Same goes for `I` (20).

A translation system works in the following stages:

```
1. [я люблю следовательно я существую] # tokenize sentence into words
2. [10 11 12 10 13]                    # look up words in the input dictionary
3. [black box]                         # machine learning system magic
4. [20 21 22 20 23]                    # look up numbers in the output dictionary
5. [I love therefore I am]             # detokenize the tokens back into a sentence
```

The first two and the last two are each combined so that we get 3 stages:

1. Encoding input: break input text into tokens, create a vocabulary of these tokens and remap each token into a number
2. Generating translation: Take input numbers, process them and return output numbers
3. Decoding output: Take output numbers, look them up in the target language dictionary and convert to text, and finally merge the converted tokens into the translated sentence.

## Tokenization

Early systems tokenized sentences into words and punctuation. But since many languages have hundreds of thousands of words it is very taxing to work with huge vocabularies. 

As of 2020 there are quite a few different versions of tokenizers, but most of the recent ones are based on sub-word tokenization - that is instead of breaking the input text into words, it breaks them down into word segments and letters. 

Let's see how this approach helps to save space. If we have an input vocabulary of 6 common words: go, going, speak, speaking, sleep, sleeping - with word-level tokenization we end up with 6 tokens. However, if we break these down into: go, go-ing, speak, speak-ing, then we have only 4 tokens in our vocabulary: go, speak, sleep, ing. That's a huge saving. 

Another important advantage is when dealing with unseen words, that aren't in our vocabulary. For example, let's say our system encounters the word 'grokking' (*), which can't be found in its vocabulary. If we split it into `grokk'-'ing', then the system might not know what to do with the first part of the word, but it gets a useful insight that 'ing' indicates a continuous tense, so it'll be able to produce a better translation.

* to grok was coined in 1961 by Robert A. Heinlein in "Stranger in a Strange Land": understand (something) intuitively or by empathy.

There are many other nuances to why the modern tokenization approach is much more superior than simple word tokenization, which won't be covered in this scope. Most of these systems are very complex to how they do the tokenization, as compared to the simple example of splitting `ing` endings that was just demonstrated.


## Tokenizer porting

The first step was to port the encoder part of the tokenizer. The decoder part won't be needed until the very end.

### fairseq's tokenizer workings

Let's understand how fairseq's tokenizer works.

fairseq uses the [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) method (BPE) for tokenization. 

* note: from here on when I refer to fairseq, I refer to the specific   [implementation](https://github.com/pytorch/fairseq/tree/master/examples/wmt19) - the project itself has dozens of different implementations of different models.

Let's see what it means:

```
import torch
sentence = "Machine Learning is great"
checkpoint_file='model4.pt'
model = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru', checkpoint_file=checkpoint_file, tokenizer='moses', bpe='fastbpe')

# encode step by step
tokens = model.tokenize(sentence)
"tokenize ", len(tokens), "-".join(tokens)
bpe = model.apply_bpe(tokens)
bpe_str = str(bpe).split()
"apply_bpe: ", len(bpe_str), bpe_str
bin = model.binarize(bpe)
"binarize: ", len(bin), bin

# compare to model.encode - should give us the same output
expected = model.encode(sentence)
"encode:   ", len(expected), expected
```

gives us:

```
('tokenize ', 25, 'M-a-c-h-i-n-e- -L-e-a-r-n-i-n-g- -i-s- -g-r-e-a-t')
('apply_bpe: ', 6, ['Mach@@', 'ine', 'Lear@@', 'ning', 'is', 'great'])
('binarize: ', 7, tensor([10217,  1419,     3,  2515,    21,  1054,     2]))
('encode:   ', 7, tensor([10217,  1419,     3,  2515,    21,  1054,     2]))
```

You can see that `model.encode` does `tokenize+apply_bpe+binarize` - we get the same output. 

The steps were:
1. tokenize: this tokenizer split the 4-word sentence into letters - 25 in total
2. apply_bpe: bpe merged the letters into words and sub-words according to its `bpecodes` file supplied by the tokenizer - we get 6 BPE chunks
3. binarize: this simply remaps the bpe chunks from the previous step into their corresponding ids in the vocabulary (which is also downloaded with the model)

You can refer to [this notebook](./nbs/tokenizer.ipynb) to see more details.

This is a good time to look inside the `bpecodes` file. Here is the top of the file:

```
# head -15 ~/porting/pytorch_fairseq_model/bpecodes
e n</w> 1423551864
e r 1300703664
e r</w> 1142368899
i n 1130674201
c h 933581741
a n 845658658
t h 811639783
e n 780050874
u n 661783167
s t 592856434
e i 579569900
a r 494774817
a l 444331573
o r 439176406
th e</w> 432025210
[...]
```

The top entries of this file include very frequent short codes. As we will see in a moment the bottom includes the most common multi-letter coders and even full long words. 

A special token `</w>` indicates that the end of the word. So in the few lines above we find:
```
e n</w> 1423551864
e r</w> 1142368899
th e</w> 432025210
```
If the second column doesn't include `</w>`, it means that it's found in the middle of the word and not the end of it.

The last column informs us of how many times this BPE code has been encountered, and this file is sorted by this column - so the most common BPE codes are on top.

By looking at the counts we now know that when this tokenizer was trained it encountered 1,423,551,864 words ending in `en`, 1,142,368,899 words ending in `er` and 432,025,210 words ending in `the`. For the latter it most likely means the actual word `the`, but it would also include words like `lathe`, `loathe`, `tithe`, etc.

This also immediately tells you that this tokenizer was trained on an enormous amount of text!

If we look at the bottom of the file:

```
# tail -10 ~/porting/pytorch_fairseq_model/bpecodes
4 x 109019
F ische</w> 109018
sal aries</w> 109012
e kt 108978
ver gewal 108978
Sten cils</w> 108977
Freiwilli ge</w> 108969
doub les</w> 108965
po ckets</w> 108953
Gö tz</w> 108943
```
we see complex combinations of sub-words which are still pretty frequent, e.g. `sal aries` for 109,012 times! so it got its own dedicated entry in the `bpecodes` map file.

How does `apply_bpe` does its work? By looking up the various combinations of letters in the `bpecodes` map file and when finding the most complex entry that fits it uses that. Going back to our example, we saw that it split `Machine` into: `Mach@@` + `ine` - let's check:

```
# grep -i ^mach  ~/porting/pytorch_fairseq_model/bpecodes
mach ine</w> 463985
Mach t 376252
Mach ines</w> 374223
mach ines</w> 214050
Mach th 119438
```
You can see that it has `mach ine</w>`. We don't see `Mach ine` in there - so it must be handling lower cased look ups when normal case is not matching.

Now let's check: `Lear@@` + `ning`

```
# grep -i ^lear  ~/porting/pytorch_fairseq_model/bpecodes
lear n</w> 675290
lear ned</w> 505087
lear ning</w> 417623
```
In there you have it `lear ning</w>` is there (again the case is not the same).

Hopefully, you can now see how this works.

One confusing thing is that if you remember the `apply_bpe` output was:
```
('apply_bpe: ', 6, ['Mach@@', 'ine', 'Lear@@', 'ning', 'is', 'great'])
```
Instead of marking endings of the words with `</w>`, it leaves those as is, but instead marks words that were not the endings with `@@`. This is probably so, because `fastBPE` implementation is used by fairseq and that's how it does things. We will make these things consistent during porting since we don't use `fastBPE`.

### Porting tokenizer's encoder to transformers

`transformers` can't rely on [`fastBPE`](https://github.com/glample/fastBPE) since the latter requires a C-compiler, but luckly someone already implemented a python version of the same in [`tokenization_xlm.py`](https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_xlm.py)

So I just copied it to `src/transformers/tokenization_fsmt.py` and with very few changes I had a working encoder part of the tokenizer. 

## Model porting


## Tokenizer decoder porting


## Optimizations, etc.

### much later

how big are embeddings? some we don't want to save them

```
model = FSMTForConditionalGeneration.from_pretrained(mname)

import torch
state_dict = model.state_dict()
torch.save(state_dict["model.encoder.embed_positions.weights"], "output")

```

### model cards


## Notes

### Autoprint all in Jupyter Notebook

My jupyter notebook is configured to automatically print all expressions, so i don't have to explicitly `print()` them (the default behavior is to print only the last expression). So if you read the outputs in my notebooks they may not the be same as if you were to run them yourself.

You can achieve the same by adding to `~/.ipython/profile_default/ipython_config.py` (create it if you don't have one):

```
c = get_config()
# Run all nodes interactively
c.InteractiveShell.ast_node_interactivity = "all"
# restore to the original behavior
# c.InteractiveShell.ast_node_interactivity = "last_expr"
```
and restarting your jupyter notebook server.

### Links to github versions of files

In order to ensure that links work when you read this article much later, the links were made to a specific version of the code and not necessarily the latest version. This is so that if files were renamed or removed you will still find the code this article is referring to. If you want to ensure you're looking at the latest version of the code, replace the hash code in the link with `master`, e.g. replace:
```
https://github.com/huggingface/transformers/blob/129fdae04033fe4adfe013b734deaec6ec34ae2e/src/transformers/modeling_fsmt.py
```
with:
```
https://github.com/huggingface/transformers/blob/master/src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py
```
