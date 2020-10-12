# pubmed

XXX: this is from an old project that used fastai - so needs to be brought up to date.

If you want to build the datasets from scratch, run [`extract.ipynb`](extract.ipynb) to download data and extract things into csv files. This will take some hours.

The nb already contains the code to download the ftp dump from which the extraction is made, but if you want to do it manually, do:

get all *gz under:
- ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/ (this is updated once a year)
- ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/ (these are the updates since baseline)
