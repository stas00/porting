# Porting pegasus evaluation datasets

`transformers` has ported the [`pegasus`](https://github.com/google-research/pegasus) project. For usage please see: [Pegasus](https://huggingface.co/transformers/model_doc/pegasus.html).

This sub-repo contains links to evaluation datasets and also build scripts that built that data.

# Datasets

Datasets that we managed to build successfully and links to s3:

dataset | 3 splits | test split
--------|------|-----
aeslc | [full](https://cdn-datasets.huggingface.co/summarization/pegasus_data/aeslc.tar.gz) | [test-only](https://cdn-datasets.huggingface.co/summarization/pegasus_data/aeslc-test.tar.gz)
arxiv | [full](https://cdn-datasets.huggingface.co/summarization/pegasus_data/arxiv.tar.gz) | [test-only](https://cdn-datasets.huggingface.co/summarization/pegasus_data/arxiv-test.tar.gz)
billsum | [full](https://cdn-datasets.huggingface.co/summarization/pegasus_data/billsum.tar.gz) | [test-only](https://cdn-datasets.huggingface.co/summarization/pegasus_data/billsum-test.tar.gz)
cnn_dailymail | [full](https://cdn-datasets.huggingface.co/summarization/pegasus_data/cnn_dailymail.tar.gz) | [test-only](https://cdn-datasets.huggingface.co/summarization/pegasus_data/cnn_dailymail-test.tar.gz)
gigaword | [full](https://cdn-datasets.huggingface.co/summarization/pegasus_data/gigaword.tar.gz) | [test-only](https://cdn-datasets.huggingface.co/summarization/pegasus_data/gigaword-test.tar.gz)
multi_news | [full](https://cdn-datasets.huggingface.co/summarization/pegasus_data/multi_news.tar.gz) | [test-only](https://cdn-datasets.huggingface.co/summarization/pegasus_data/multi_news-test.tar.gz)
newsroom | [full](https://cdn-datasets.huggingface.co/summarization/pegasus_data/newsroom.tar.gz) | [test-only](https://cdn-datasets.huggingface.co/summarization/pegasus_data/newsroom-test.tar.gz)
pubmed | [full](https://cdn-datasets.huggingface.co/summarization/pegasus_data/pubmed.tar.gz) | [test-only](https://cdn-datasets.huggingface.co/summarization/pegasus_data/pubmed-test.tar.gz)
reddit_tifu | [full](https://cdn-datasets.huggingface.co/summarization/pegasus_data/reddit_tifu.tar.gz) | [test-only](https://cdn-datasets.huggingface.co/summarization/pegasus_data/reddit_tifu-test.tar.gz)
wikihow | [full](https://cdn-datasets.huggingface.co/summarization/pegasus_data/wikihow.tar.gz) | [test-only](https://cdn-datasets.huggingface.co/summarization/pegasus_data/wikihow-test.tar.gz)
xsum | [full](https://cdn-datasets.huggingface.co/summarization/pegasus_data/xsum.tar.gz) | [test-only](https://cdn-datasets.huggingface.co/summarization/pegasus_data/xsum-test.tar.gz)

Each *full* archive includes the following files:

```
test.source
test.target
train.source
train.target
validation.source
validation.target
```

Each *test-only* archive includes just:

```
test.source
test.target
```


Datasets that we couldn't figure out:

* big_patent - we couldn't build this dataset, see https://github.com/google-research/pegasus/issues/114

For history purposes, here is the [issue](https://github.com/huggingface/transformers/issues/7647) where the process has been discussed.



# Building data from scratch

Should you want to build from scratch, use these notes.

Currently the datasets are pulled from either `datasets` or `tfds` or `tfds_transformed`. 

For each dataset you will find a folder with `process.txt` that includes instructions on how to build it.

The top-level `process-all.py` that builds most of them at once will only work once each was built via its folder's `process.txt`. This is because many of the datasets require a one-time manual download/tinkering.

Most build scripts use `pegasus` which takes a bit of tinkering to install:

```
git clone https://github.com/google-research/pegasus
cd pegasus
pip install pygame==2.0.0.dev12
perl -pi -e 's|tensorflow-text==1.15.0rc0|tensorflow-text|; s|tensor2tensor==1.15.0|tensor2tensor|; s|tensorflow-gpu==1.15.2|tensorflow-gpu|' requirements.txt setup.py
pip install -r requirements.txt
pip install -e .
```
Then you will also need:
```
pip install tensorflow_datasets -U
pip install datasets
```

# Evaluation

Each sub-folder's `process.txt` contains the command to run the evaluation. It assumes you have already installed `transformers` with its prerequisites:

```
git clone https://github.com/huggingface/transformers/
cd transformers
pip install -e .[dev]
pip install -r examples/requirements.txt    
```
And finally:
```
cd ./examples/seq2seq
```
as that's where the eval scripts are located.

see `README.md` inside `examples/seq2seq` for additional information about eval scripts.


# Problems

If you encounter any problems with building eval data, please create an issue here. If you have any issue with outcomes this is an issue for [`transformers`](https://github.com/huggingface/transformers/issues).

If you manage to figure out how to build `big_patent`, see this [issue](https://github.com/google-research/pegasus/issues/114) that would be amazing! Thank you!

## Authors

This area is a collaboration of @sshleifer, @patil-suraj and @stas00 (and more contributors are welcome)


