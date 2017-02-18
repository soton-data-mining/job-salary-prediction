#!/usr/bin/env python
import re

import nltk
from collections import Counter
from data_extractor.data_getter import DataGetter


# ensure these are downloaded on import.
nltk.download('wordnet')
nltk.download('punkt')


def pre_process_job_titles():
    """
    Process the list of job titles in the CSV training data, doing minor modifications to it,
    namely, remove cities and stop words.
    :return: A string with each word separated by a space for the most part. slashes may live.
    """
    job_titles = DataGetter.get_raw_training_data("Title")
    cached_stop_words = set(DataGetter.get_stop_word_inc_cities())  # set is faster in loop below
    job_title_processed = []
    for job_title in job_titles:
        title = nltk.word_tokenize(job_title)
        removed_stop_title = " ".join([word.lower() for word in title if word.lower() not in cached_stop_words])
        job_title_processed.append(removed_stop_title)
    return job_title_processed


def process_actual_jobs_list():
    """
    Returns a list of unique, stemmed, official-ish job titles and roles from the
    UK National Archives' list of job roles.
    :return: a tuple with two sets in it (roles, modifiers) where a role is the main
    job role (account, engin, someth) and modifiers are (train, work, etc)
    """
    stemmer = nltk.stem.snowball.EnglishStemmer()
    unprocessed_job_roles = DataGetter.get_unique_job_roles()

    set_of_unique_jobs = set()
    set_of_unique_modifiers = set()

    for job in unprocessed_job_roles:
        # stem the jobs with
        job_parts = [stemmer.stem(job_part.strip()) for job_part in job.split(",")]
        set_of_unique_jobs.add(job_parts[0])
        for word in job_parts[1:]:
            set_of_unique_modifiers.add(word)

    # remove any overlap in modifiers:
    unique_modifiers_without_roles = set_of_unique_modifiers.difference(set_of_unique_jobs)

    return set_of_unique_jobs, unique_modifiers_without_roles


def _get_stemmed_sorted_role_and_modifiers(job_title_text,
                                           official_unique_jobs,
                                           official_unique_modifiers,
                                           stemmer):
    """
    Does the dirty work involved in creating an ordered stemmed sentence that fairly cleanly
    identifies a job title. basically, two titles like:
    "Old people nurses" and "nurse for old people" should both come out to "nurs old peopl"
    or similar (haven't tested that exact sentence)

    :param job_title_text: the job title we'd like to convert in to a sorted stemmed sentence
    :param official_unique_jobs: the set of unique job titles (already stemmed and lowercase)
    :param official_unique_modifiers: the set of unique job modifiers (already stemmed and
                                      lowercase)
    :param stemmer: the stemmer to be used (on which the .stem method is called, so handle with
                    care)
    :return: two strings containing the sorted stemmed sentence for title and
             modifier (title, modifiers)
    """
    # do some preprocessing on the job title we're searching for.
    # 0) lowercase it all
    # 1) split in to text blocks
    # 2) stem it
    # 3) pick out only those we like from the 'official' list.

    individual_words = nltk.word_tokenize(job_title_text.lower())
    stemmed_words = [stemmer.stem(word) for word in individual_words]  # lemmatize each word

    matching_jobs = [
        title for title in official_unique_jobs if title in stemmed_words
    ]
    matching_modifiers = [
        modifier for modifier in official_unique_modifiers if modifier in stemmed_words
    ]

    matching_jobs.sort()
    matching_modifiers.sort()

    return matching_jobs, matching_modifiers


def get_stemmed_sorted_role_and_modifiers(title):
    """
    Convenience method so you don't have to build a brazillian things just to check one stemmed
    title, please don't use this in a loop - it initializes everything :(
    :param title: the title you'd like the stemmed sentence of
    :return: the stemmed matching roles and titles
    """
    # get official roles and modifiers sets:
    official_unique_jobs, official_unique_modifiers = process_actual_jobs_list()
    # set up our english snowball stemmer
    stemmer = nltk.stem.snowball.EnglishStemmer()

    return _get_stemmed_sorted_role_and_modifiers(
        title,
        official_unique_jobs,
        official_unique_modifiers,
        stemmer
    )


def get_stemmed_sentences(processed_job_titles):
    """
    get two vectors which are the processed job titles and modifiers to use for ML
    :return: Vectors with the processed/stemmed datas.
    """

    # get official roles and modifiers sets:
    official_unique_jobs, official_unique_modifiers = process_actual_jobs_list()
    # set up our english snowball stemmer
    stemmer = nltk.stem.snowball.EnglishStemmer()

    # prep some mappings
    mapped_titles = []
    mapped_modifiers = []

    for job_title_text in processed_job_titles:
        sorted_job_stem, sorted_mod_stem = _get_stemmed_sorted_role_and_modifiers(
            job_title_text,
            official_unique_jobs,
            official_unique_modifiers,
            stemmer
        )

        mapped_titles.append(' '.join(sorted_job_stem))
        mapped_modifiers.append(' '.join(sorted_mod_stem))

    return mapped_titles, mapped_modifiers

if __name__ == "__main__":
    processed_job_titles = pre_process_job_titles()
    job_titles, mods = get_stemmed_sentences(processed_job_titles)
    jobCounter = Counter(job_titles)
    modCounter = Counter(mods)


    print(jobCounter)
