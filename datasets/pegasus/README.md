# Porting pegasus datasets

[Issue 7647](https://github.com/huggingface/transformers/issues/7647)

This sub-repo contains scripts that will quickly build eval data for pegasus.

# Status

Datasets that we were successful at bulding:

* xsum
* cnn_dailymail
* newsroom
* multi_news
* gigaword
* wikihow
* reddit_tifu
* arxiv
* pubmed
* aeslc
* billsum

Datasets that we couldn't figure out:

* big_patent - couldn't build this dataset, see https://github.com/google-research/pegasus/issues/114

# Getting data

Currently the datasets are pulled from either `datasets` or `tfds` or `tfds_transformed`. 

For each dataset you will find a folder with `process.txt` that includes instructions on how to build it.

The top-level `process-all.py` that builds most of them at once will only work once each was built via its folder. This is because many of them require some manual tinkering.

If you encounter any problems with building eval data, please create an issue here. If you have any issue with outcomes this is an issue for [`transformers`](https://github.com/huggingface/transformers/issues).

If you manage to figure out how to build `big_patent`, see this [issue](https://github.com/google-research/pegasus/issues/114) that would be amazing! Thank you!

## Authors

This area is a collaboration of @sshleifer, @patil-suraj and @stas00 (and more contributors are welcome)


