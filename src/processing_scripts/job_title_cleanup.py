#!/usr/bin/env python

import csv
import re
import sys

from nltk.tokenize.treebank import TreebankWordTokenizer


regex_list = [
    # preceding/trailing whitespace eliminated via strip()
    r'x{0,1}\*\*\*\*[sdtxyr]{,2} *',  # gets rid of ****yr, ****, ****x
]

crap_list = ['•', '`']

_compiled_regexes = [
    re.compile(regex, re.IGNORECASE) for regex in regex_list
]


def get_all_job_titles(filename):
    job_titles = []
    with open(filename, 'r') as rawfile:
        csvreader = csv.reader(rawfile)
        for line in csvreader:
            job_titles.append(line[1])
    return job_titles


def get_unique_job_titles(filename):
    return set(get_all_job_titles(filename))


def clean_title(title):
    """
    This method is built to take in a job title and churn out a 'clean' title
    that can be used for further processing, it should get rid of filth like:

    - CAPITAL LETTERS
    - censored ****'s in the data
    - censored ****
    - preceding/trailing whitespace
    - crappy symbols

    :param title: the job title we need to clean out.
    :return: a hopefully clean job title, possibly corrected for typos
    """

    title.strip()  # make regexes easier to match
    title = title.lower()

    for crap in crap_list:
        # eliminate awkward characters, usually eliminated by tokenization
        # but why not?
        title = title.replace(crap, '')

    for regx in _compiled_regexes:
        # replace anything matching a regex with the void.
        title = re.sub(regx, '', title)

    # TODO: typo correction (word distance)

    title.strip()  # drop preceding/trailing whitespace post-regex
    return title


if __name__ == '__main__':
    """
    Demonstration only, probs not used normally
    """
    filename = sys.argv[1]

    job_titles = get_unique_job_titles(filename)
    clean_titles = [clean_title(title) for title in job_titles]
    clean_titles.sort()

    tokker = TreebankWordTokenizer()
    tokenized_job_titles = [tokker.tokenize(title) for title in job_titles]

    for token_list in tokenized_job_titles:
        print(token_list)
