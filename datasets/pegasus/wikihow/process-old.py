#!/usr/bin/env python

import os
import glob
import json
import re

def get_art_abs_wikihow(path):
    articles = sorted(glob.glob('%s/*' % path))
    for a in articles:
        try:
            with open(a, 'r') as f:
                text = f.read()
            splits = text.split('@article')
            
            abstract = splits[0].replace('@summary\n', '')
            #abstract = re.sub(r'\\', r'\\\\', abstract)
            abstract = "<n>".join(l.strip().capitalize() for l in abstract.split("\n"))
            abstract = re.sub(r'(<n>)+', '<n>', abstract)
            abstract = re.sub(r'<n>$', '', abstract)
            abstract = re.sub(' +', ' ', abstract).strip()
            
            #article = splits[1].replace('\n', ' ').replace('@article', '').strip()
            article = splits[1].replace('@article\n', '').strip()
            article = re.sub(r'\n$', '', article)
            #article = re.sub(r'\\', r'\\\\', article)
            article = re.sub(r'\n', '<n>', article)
            #article  = article.replace("\n", "<n>")
            article = re.sub(' +', ' ', article).strip()
                
            yield article, abstract
        except Exception as e:
            yield None

def write_to_bin(lines, out_prefix):
    print("Making bin file for %s..." % out_prefix)

    with open(out_prefix + '.source', 'wt') as source_file, open(out_prefix + '.target', 'wt') as target_file:
        for idx, line in enumerate(lines):
            if idx % 10000 == 0:
                print("Writing story %i" % idx)

            # Get the strings to write to .bin file
            if line == None: continue
            article, abstract = line
            article_len, abstract_len = len(article), len(abstract)
            

            #print(f"article {article}")
            #print(f"abstract {abstract}")
            #  a threshold is used to remove short articles with long summaries as well as articles with no summary
            if article_len and abstract_len and abstract_len < 0.75*article_len:
                # remove extra commas in abstracts
                #abstract = abstract.replace(".,",".")
                # remove extra commas in articles
                #article = article.replace(";,", "")
                #article = article.replace(".,",".")
                #article = re.sub(r'[.]+[\n]+[,]',".\n", article)
                
                # truncate final \n
                #article = re.sub(r'\n$', '', article)

                #abstract = abstract.strip().replace("\n", "")
                #article  = article.replace("\n", "<n>")
                #article = re.sub(' +', ' ', article).strip()
                
                article  = re.sub(r'[\r\n]', '<n>', article)
                abstract = re.sub(r'[\r\n]', '<n>', abstract)
                
                # Write article and abstract to files
                source_file.write(article + '\n')
                target_file.write(abstract + '\n')

    print("Finished writing files")

def create_stories(save_path='data'):
    os.makedirs(save_path, exist_ok=True)

    lines = get_art_abs_wikihow('./test_articles')
    write_to_bin(lines, os.path.join(save_path, "test"))

    # lines = get_art_abs_wikihow('./valid_articles')
    # write_to_bin(lines, os.path.join(save_path, "val"))

    # lines = get_art_abs_wikihow('./train_articles')
    # write_to_bin(lines, os.path.join(save_path, "train"))

create_stories()
